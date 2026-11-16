"""Local pipe worker. Imports only public OpenPI and its inference dependencies."""

import argparse
import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def restore_jax_params(path):
    """Public Orbax restore API; adapt StepMetadata without changing OpenPI source."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import traverse_util

    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("x",)), jax.sharding.PartitionSpec()
    )
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(path)
        item_metadata = metadata.item_metadata
        item = {"params": item_metadata["params"]}
        params = reader.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=jax.Array, dtype=jnp.bfloat16
                    ),
                    item,
                ),
            ),
        )["params"]
    flat = traverse_util.flatten_dict(params)
    if flat and all(key[-1] == "value" for key in flat):
        flat = {key[:-1]: value for key, value in flat.items()}
    return traverse_util.unflatten_dict(flat)


def build(model_path, manifest, task):
    import jax
    import numpy as np
    import sentencepiece
    from openpi import transforms
    from openpi.models import pi0_config, tokenizer
    from openpi.policies.policy import Policy
    from openpi.shared import normalize

    devices = jax.devices()
    if not devices or any(d.platform != "gpu" for d in devices):
        raise RuntimeError("OpenPI JAX requires CUDA GPU devices; CPU fallback is forbidden")

    class LocalTokenizer(tokenizer.PaligemmaTokenizer):
        def __init__(self):
            self._max_len = 200
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(model_path / manifest["tokenizer"])
            )

    norm_stats = normalize.load((model_path / manifest["norm_stats"]).parent)
    if set(norm_stats) != {"state", "actions"}:
        raise ValueError("JAX normalization must contain exactly state and actions")
    for stats in norm_stats.values():
        for key in ("mean", "std", "q01", "q99"):
            values = np.asarray(getattr(stats, key))
            if values.shape != (16,) or not np.isfinite(values).all():
                raise ValueError("JAX normalization must describe the physical 16-D layout")

    def inputs(data):
        state = data["state"]
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("invalid JAX observation state")
        names = {"head": "base_0_rgb", "left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}
        images = {}
        for name, target in names.items():
            image = data[name]
            if image.shape != (168, 224, 3) or image.dtype != np.uint8:
                raise ValueError("JAX input images must be RGB uint8 [168,224,3]")
            images[target] = image
        return {
            "state": state,
            "image": images,
            "image_mask": {key: np.True_ for key in images},
            "prompt": task,
        }

    def outputs(data):
        return {"actions": data["actions"][..., :16]}

    config = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200)
    params = restore_jax_params(model_path / "params")
    policy = Policy(
        config.load(params, remove_extra_params=False),
        transforms=[
            inputs,
            transforms.Normalize(norm_stats, use_quantiles=True),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(LocalTokenizer(), discrete_state_input=True),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[transforms.Unnormalize(norm_stats, use_quantiles=True), outputs],
        sample_kwargs={"num_steps": 10},
    )
    return policy, {
        "jax": jax.__version__,
        "devices": [str(d) for d in devices],
        "platforms": [d.platform for d in devices],
    }


def main():
    import resource

    _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard_limit))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    wire = sys.stdout

    def reply(value):
        wire.write(json.dumps(value, allow_nan=False) + "\n")
        wire.flush()

    try:
        with redirect_stdout(sys.stderr):
            manifest = json.loads((args.model / "onero_openpi.json").read_text())
            policy, identity = build(args.model, manifest, args.task)
        reply({"ready": True, **identity})
        while line := sys.stdin.buffer.readline(8 * 1024 * 1024 + 1):
            if len(line) > 8 * 1024 * 1024 or not line.endswith(b"\n"):
                raise ValueError("oversized or incomplete worker request")
            request = json.loads(line)
            if request.get("op") == "close":
                return
            if request.get("op") != "infer":
                raise ValueError("unknown worker request")
            import numpy as np

            payload = base64.b64decode(request["observation"], validate=True)
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                obs = {name: arrays[name] for name in ("state", "head", "left", "right")}
            with redirect_stdout(sys.stderr):
                actions = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
            if actions.shape != (50, 16) or not np.isfinite(actions).all():
                raise ValueError("invalid JAX action chunk")
            reply({"actions": actions.tolist()})
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        reply({"error": str(exc)})
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
