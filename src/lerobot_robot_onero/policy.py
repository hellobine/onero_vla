"""Small adapter around upstream public policy and processor APIs."""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .images import model_image
from .protocol import FEATURES, validate_action
from .release import runtime_identity


def model_contract(path: Path):
    config = json.loads((path / "config.json").read_text())
    contract = json.loads((path / "onero_model.json").read_text())
    if contract.get("schema_version") != 1:
        raise ValueError("onero_model.json: unsupported schema_version")
    if type(contract.get("test_only", False)) is not bool:
        raise ValueError("onero_model.json: test_only must be boolean")
    for key in ("state_features", "action_features"):
        if contract.get(key) != list(FEATURES):
            raise ValueError(f"onero_model.json: {key} must match the physical 16-D order")
    if contract.get("action_representation") != "absolute_joint_positions":
        raise ValueError("only absolute joint-position models are supported")
    if contract.get("units") != "radians_and_normalized_grippers":
        raise ValueError("model contract units are incompatible")
    camera_map = contract.get("camera_map", {})
    if set(camera_map) != {"head", "left", "right"} or len(set(camera_map.values())) != 3:
        raise ValueError("model contract must map three distinct cameras")
    if any(
        not isinstance(v, str) or not v.startswith("observation.images.")
        for v in camera_map.values()
    ):
        raise ValueError("model camera keys must start with observation.images.")
    inputs, outputs = config.get("input_features", {}), config.get("output_features", {})
    state_shape = inputs.get("observation.state", {}).get("shape")
    internal_pi05 = config.get("type") == "pi05" and state_shape == [32]
    if state_shape != [16] and not internal_pi05:
        raise ValueError("model state must be 16-D or the declared PI05 32-D internal layout")
    if internal_pi05 and config.get("action_feature_names") != list(FEATURES):
        raise ValueError(
            "PI05 internal-state layout requires the explicit external 16-D action order"
        )
    if config.get("use_relative_actions", False):
        raise ValueError("relative-action checkpoints are not supported")
    if outputs.get("action", {}).get("shape") != [16] or set(outputs) != {"action"}:
        raise ValueError("model action must have shape [16]")
    if set(inputs) != {"observation.state", *camera_map.values()}:
        raise ValueError("checkpoint image keys do not match the declared three-camera contract")
    for key in camera_map.values():
        shape = inputs[key].get("shape", [])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"{key} must declare a 3-channel image")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("local model directory contains no safetensors weights")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (path / name).is_file():
            raise ValueError(f"missing local processor: {name}")
    return config, contract


def model_identity(path):
    digest = hashlib.sha256()
    files = []
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("model directory must contain materialized files, not symlinks")
        if file.is_file():
            relative = file.relative_to(path).as_posix()
            with file.open("rb") as stream:
                file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            digest.update(f"{relative}\0{file_hash}\n".encode())
            files.append({"path": relative, "sha256": file_hash, "bytes": file.stat().st_size})
    return {"sha256": digest.hexdigest(), "files": files}


class LocalPolicy:
    def __init__(self, settings):
        runtime_identity()
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.processor import policy_output_to_transition, transition_to_policy_action

        self.torch = torch
        self.settings = settings
        self.prepare = prepare_observation_for_inference
        self.action_from_output = lambda output: transition_to_policy_action(
            policy_output_to_transition(output)
        )
        _metadata, self.contract = model_contract(settings.model_path)
        self.identity = model_identity(settings.model_path)
        self.identity["backend"] = "lerobot_torch"
        self.identity["transport_image_resolution"] = settings.image_resolution
        if settings.tokenizer_path is not None:
            self.identity["tokenizer"] = model_identity(settings.tokenizer_path)
        config = PreTrainedConfig.from_pretrained(settings.model_path, local_files_only=True)
        config.device = settings.device
        if settings.compile_model is not None:
            if not hasattr(config, "compile_model"):
                raise ValueError("this policy has no compile_model setting")
            config.compile_model = settings.compile_model
        # A full checkpoint already includes its vision weights. Avoid torchvision's
        # separate initialization download, which is not governed by HF offline flags.
        if hasattr(config, "pretrained_backbone_weights"):
            config.pretrained_backbone_weights = None
        policy_class = get_policy_class(config.type)
        if config.type == "pi05":
            from .strict_load import load_pi05_strict

            self.policy = load_pi05_strict(policy_class, config, settings.model_path)
        else:
            self.policy = policy_class.from_pretrained(
                settings.model_path, config=config, local_files_only=True, strict=True
            )
        self.policy.to(settings.device).eval()
        pre_overrides = {"device_processor": {"device": settings.device}}
        if settings.tokenizer_path is not None:
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": str(settings.tokenizer_path)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(settings.model_path),
            preprocessor_overrides=pre_overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        for processor in (self.preprocessor, self.postprocessor):
            for step in processor.steps:
                if type(step).__name__ in (
                    "RelativeActionsProcessorStep",
                    "AbsoluteActionsProcessorStep",
                ) and getattr(step, "enabled", True):
                    raise ValueError(
                        "relative-action pipelines need a separately qualified runtime"
                    )
        self.reset()

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _batch(self, raw):
        inputs = {"observation.state": np.asarray([raw[k] for k in FEATURES], dtype=np.float32)}
        for camera, key in self.contract["camera_map"].items():
            inputs[key] = model_image(raw[camera], self.settings.image_resolution)
        return self.preprocessor(
            self.prepare(
                inputs,
                self.torch.device(self.settings.device),
                self.settings.task,
                f"onero_{self.settings.robot.model}",
            )
        )

    def _autocast(self):
        return (
            self.torch.autocast(device_type="cuda")
            if self.settings.device.startswith("cuda") and self.policy.config.use_amp
            else nullcontext()
        )

    def select(self, raw):
        torch = self.torch
        with torch.inference_mode(), self._autocast():
            output = self.policy.select_action(self._batch(raw))
            action = self.postprocessor(self.action_from_output(output))
        action = action.detach().to(device="cpu", dtype=torch.float32).numpy()
        if action.shape != (1, 16) or not np.isfinite(action).all():
            raise ValueError(
                f"policy returned invalid absolute action shape/values: {action.shape}"
            )
        return validate_action(dict(zip(FEATURES, action[0], strict=True)))

    def validate_chunk_mode(self):
        from lerobot.policies import ExecutionMode

        spec = self.policy.chunk_inference_spec()
        if not spec.current_observation_only or ExecutionMode.CHUNK not in spec.modes:
            raise ValueError("this policy does not support current-observation chunk inference")
        if spec.execution_steps / self.settings.fps > self.settings.processing.max_chunk_age_s:
            raise ValueError("chunk execution horizon exceeds max_chunk_age_s")
        return spec

    def predict_chunk(self, raw):
        torch = self.torch
        spec = self.validate_chunk_mode()
        with torch.inference_mode(), self._autocast():
            output = self.policy.predict_action_chunk(self._batch(raw))
            actions = self.action_from_output(output)
            if tuple(actions.shape) != (1, spec.prediction_steps, 16):
                raise ValueError("policy chunk shape does not match its declared [1,T,16] contract")
            canonical = self.postprocessor(actions)
        values = canonical.detach().to(device="cpu", dtype=torch.float32).numpy()
        if values.shape != (1, spec.prediction_steps, 16) or not np.isfinite(values).all():
            raise ValueError("policy/processor returned an invalid canonical action chunk")
        return values[0, : spec.execution_steps].copy()
