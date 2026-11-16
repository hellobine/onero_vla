"""GPU/optional receive-only ROS acceptance with an explicitly synthetic ACT model.

No motion-enabled Robot is ever created. The generated weights are test data.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from lerobot_robot_onero.cli import preflight
from lerobot_robot_onero.config import OneroRobotConfig, OneroRosCameraConfig
from lerobot_robot_onero.policy import LocalPolicy
from lerobot_robot_onero.protocol import FEATURES
from lerobot_robot_onero.release import runtime_identity
from lerobot_robot_onero.settings import Settings, configure_environment, load_settings


def smoke(output, device, config=None):
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "synthetic-act-model"
    model_dir.mkdir()
    cameras = {n: OneroRosCameraConfig(width=32, height=32) for n in ("head", "left", "right")}
    settings = (
        dataclasses.replace(load_settings(config), model_path=model_dir, device=device)
        if config
        else Settings(OneroRobotConfig(cameras=cameras), model_dir, device=device)
    )
    configure_environment(settings)
    import torch
    import torchvision
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies import ACTConfig, get_policy_class, make_pre_post_processors

    report = {
        "ok": False,
        "synthetic_model": True,
        "motion_attempted": False,
        "hardware_checked": False,
        **runtime_identity(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
    }
    try:
        torch.set_num_threads(2)
        torch.manual_seed(7)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; refusing a silent CPU fallback")
            report.update(
                gpu=torch.cuda.get_device_name(0),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                compiled_architectures=torch.cuda.get_arch_list(),
            )
            x = torch.ones((64, 64), device=device)
            assert torch.isfinite(x @ x).all().item()
            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)
            assert torchvision.ops.nms(boxes, torch.ones(1, device=device), 0.5).numel() == 1
            torch.cuda.synchronize()
        model_config = ACTConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
                **{
                    f"observation.images.{n}": PolicyFeature(
                        type=FeatureType.VISUAL, shape=(3, c.height, c.width)
                    )
                    for n, c in settings.robot.cameras.items()
                },
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
            normalization_mapping={
                n: NormalizationMode.IDENTITY for n in ("STATE", "VISUAL", "ACTION")
            },
            device=device,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
            chunk_size=16,
            n_action_steps=16,
        )
        model = get_policy_class("act")(model_config)
        model.save_pretrained(model_dir)
        before, after = make_pre_post_processors(model_config)
        before.save_pretrained(model_dir)
        after.save_pretrained(model_dir)
        contract = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/onero_model.example.json").read_text()
        )
        contract["test_only"] = True
        (model_dir / "onero_model.json").write_text(json.dumps(contract, indent=2))
        del model
        engine = LocalPolicy(settings)
        raw = {
            **dict.fromkeys(FEATURES, 0.0),
            **{
                n: np.zeros((c.height, c.width, 3), dtype=np.uint8)
                for n, c in settings.robot.cameras.items()
            },
        }
        latencies = []
        for _ in range(4):
            start = time.monotonic()
            action = engine.select(raw)
            latencies.append((time.monotonic() - start) * 1000)
            assert len(action) == 16 and np.isfinite(list(action.values())).all()
        report["select_action_latency_ms"] = latencies
        report["action_shape"] = [16]
        if config:
            engine.reset()
            report["preflight"] = preflight(settings, engine, output / "real-observation")
            report["hardware_checked"] = True
        report["ok"] = True
        return report
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        (output / "smoke-result.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, help="Optional real ROS read-only configuration")
    args = parser.parse_args()
    print(json.dumps(smoke(args.output.resolve(), args.device, args.config), indent=2))
