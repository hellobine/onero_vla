"""Real upstream imports, factory discovery and CPU ACT checkpoint inference."""

import ast
import dataclasses
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_onero import OneroRobot, OneroRobotConfig, OneroRosCameraConfig
from lerobot_robot_onero.cli import main, preflight
from lerobot_robot_onero.policy import LocalPolicy, model_contract
from lerobot_robot_onero.protocol import FEATURES
from lerobot_robot_onero.settings import Settings, configure_environment
from lerobot_robot_onero.smoothing import ProcessingConfig

ROOT = Path(__file__).resolve().parents[1]


def test_real_upstream_plugin_factories(tmp_path):
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.robots.utils import make_robot_from_config

    robot = make_robot_from_config(OneroRobotConfig(calibration_dir=tmp_path))
    assert isinstance(robot, OneroRobot)
    assert not robot.is_connected
    cameras = make_cameras_from_configs({"head": OneroRosCameraConfig()})
    assert not cameras["head"].is_connected


def test_installed_plugin_is_discoverable_in_fresh_process():
    # LeRobot discovers distribution names with underscores; don't mask a packaging bug
    # by importing our package before discovery.
    script = """
import sys
from lerobot.utils.import_utils import register_third_party_plugins
assert 'lerobot_robot_onero' not in sys.modules
register_third_party_plugins()
assert 'lerobot_robot_onero' in sys.modules
assert 'rclpy' not in sys.modules
assert not any(n == 'one_vla' or n.startswith('one_vla.') for n in sys.modules)
"""
    env = {**os.environ, "PYTHONPATH": "", "HF_HUB_OFFLINE": "1"}
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=30)


def test_client_source_has_no_private_imports():
    for path in [
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "jax_runtime/onero_jax_worker").rglob("*.py"),
        *(ROOT / "jax_runtime_5090/onero_jax_worker").rglob("*.py"),
    ]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                m.split(".")[0] in {"one_vla", "one_vla_contracts", "algorithms"} for m in modules
            )


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies import ACTConfig, get_policy_class, make_pre_post_processors

    torch.set_num_threads(1)
    torch.manual_seed(7)
    path = tmp_path_factory.mktemp("act-offline")
    cameras = {
        name: OneroRosCameraConfig(width=32, height=32) for name in ("head", "left", "right")
    }
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            **{
                f"observation.images.{n}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
                for n in cameras
            },
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        normalization_mapping={
            name: NormalizationMode.IDENTITY for name in ("STATE", "VISUAL", "ACTION")
        },
        device="cpu",
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        chunk_size=2,
        n_action_steps=2,
    )
    policy = get_policy_class("act")(config)
    policy.save_pretrained(path)
    pre, post = make_pre_post_processors(config)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    (path / "onero_model.json").write_bytes(
        (ROOT / "configs/onero_model.example.json").read_bytes()
    )
    settings = Settings(OneroRobotConfig(cameras=cameras), path, device="cpu")
    configure_environment(settings)
    return settings


def observation():
    return {
        **dict.fromkeys(FEATURES, 0.0),
        **{name: np.zeros((32, 32, 3), dtype=np.uint8) for name in ("head", "left", "right")},
    }


def test_real_offline_act_load_and_select(tiny_model, monkeypatch):
    import socket

    def network_denied(*args, **kwargs):
        raise AssertionError("offline inference attempted to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", network_denied)
    assert importlib.metadata.version("lerobot") == "0.7.0"
    engine = LocalPolicy(tiny_model)
    for _ in range(3):
        action = engine.select(observation())
        assert list(action) == list(FEATURES)
        assert np.isfinite(list(action.values())).all()
    engine.reset()
    # Latest LeRobot may return a structured PolicyOutput instead of a Tensor.
    original = engine.policy.select_action
    engine.policy.select_action = lambda batch: {"action": original(batch)}
    assert len(engine.select(observation())) == 16


def test_preflight_uses_real_model_without_motion(tiny_model, tmp_path, monkeypatch):
    import lerobot_robot_onero.cli as cli

    monkeypatch.setattr(cli, "readonly_observation", lambda settings: observation())
    output = tmp_path / "evidence"
    result = preflight(tiny_model, LocalPolicy(tiny_model), output)
    assert result["ok"] is True
    assert result["executed_actions"] is False
    assert result["motion_attempted"] is False
    assert np.load(output / "predicted_action.npy").shape == (16,)
    assert json.loads((output / "result.json").read_text())["ok"]
    with pytest.raises(FileExistsError):
        preflight(tiny_model, LocalPolicy(tiny_model), output)


def test_full_checkpoint_does_not_download_backbone_initialization(tiny_model, monkeypatch):
    import socket

    path = tiny_model.model_path / "config.json"
    original = path.read_text()
    config = json.loads(original)
    config["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
    path.write_text(json.dumps(config))

    def deny_network(*args, **kwargs):
        raise AssertionError("full local checkpoint tried to download initial backbone weights")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    try:
        engine = LocalPolicy(tiny_model)
        assert len(engine.select(observation())) == 16
    finally:
        path.write_text(original)


def test_mismatched_feature_order_rejected(tiny_model):
    path = tiny_model.model_path / "onero_model.json"
    original = path.read_text()
    contract = json.loads(original)
    contract["action_features"] = list(reversed(contract["action_features"]))
    path.write_text(json.dumps(contract))
    try:
        with pytest.raises(ValueError, match="action_features"):
            model_contract(tiny_model.model_path)
    finally:
        path.write_text(original)


def test_no_hardware_cli_and_motion_optin(tiny_model, tmp_path, capsys):
    config = tmp_path / "station.yaml"
    config.write_text(f"model: {{path: '{tiny_model.model_path}', device: cpu}}")
    assert main(["check", "--config", str(config), "--no-hardware"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hardware_checked"] is False
    assert main(["run", "--config", str(config)]) == 1
    assert "--enable-motion" in capsys.readouterr().out


def test_preflight_evidence_rejects_model_subdirectory(tiny_model):
    with pytest.raises(ValueError, match="outside"):
        preflight(tiny_model, None, tiny_model.model_path / "evidence")


def test_motion_refuses_synthetic_checkpoint_before_robot_access(monkeypatch):
    import lerobot_robot_onero.cli as cli
    import lerobot_robot_onero.policy as policy

    monkeypatch.setattr(
        policy,
        "LocalPolicy",
        lambda settings: type("Test", (), {"contract": {"test_only": True}})(),
    )
    monkeypatch.setattr(
        cli, "readonly_observation", lambda settings: pytest.fail("hardware accessed")
    )
    with pytest.raises(ValueError, match="test-only"):
        from types import SimpleNamespace

        cli.run(SimpleNamespace(backend="lerobot_torch"), 1)


@pytest.mark.parametrize(
    "command,extra,expected",
    [("run", [], None), ("run", ["--duration", "5"], 5.0), ("shadow", [], 60.0)],
)
def test_cli_normal_run_is_continuous_despite_legacy_yaml_duration(
    tiny_model, tmp_path, monkeypatch, capsys, command, extra, expected
):
    import lerobot_robot_onero.cli as cli

    config = tmp_path / "station.yaml"
    config.write_text(f"model: {{path: '{tiny_model.model_path}'}}\nruntime: {{duration_s: 60}}")
    durations = []

    def capture(settings, duration, *args):
        durations.append(duration)
        return {"ok": True}

    monkeypatch.setattr(cli, "run", capture)
    monkeypatch.setattr(cli, "shadow", capture)
    options = ["--enable-motion"] if command == "run" else ["--output", str(tmp_path / "preview")]
    assert cli.main([command, "--config", str(config), *options, *extra]) == 0
    assert durations == [expected]
    assert json.loads(capsys.readouterr().out)["ok"]


@pytest.mark.parametrize("continuous", [True, False])
def test_continuous_run_passes_sixty_seconds_and_cleans_up_on_interrupt(
    tiny_model, monkeypatch, continuous, tmp_path
):
    from types import SimpleNamespace

    import lerobot_robot_onero.action_runtime as action_module
    import lerobot_robot_onero.cli as cli
    import lerobot_robot_onero.jax_policy as policy_module
    import lerobot_robot_onero.robot as robot_module

    clock = SimpleNamespace(now=0.0)
    closed = set()
    sent_at = []
    action = dict.fromkeys(FEATURES, 0.0)

    class Engine:
        contract = {"test_only": False}

        def select(self, observation):
            return action

        def reset(self):
            pass

        def close(self):
            closed.add("engine")

    class Robot:
        def __init__(self, config):
            assert not config.read_only

        def connect(self):
            pass

        def get_observation(self):
            return {}

        def send_action(self, value):
            sent_at.append(clock.now)
            return value

        def disconnect(self):
            closed.add("robot")

    class Runtime:
        def __init__(self, engine, settings):
            self.calls = 0

        def tick(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            clock.now = 61.0
            return action

        def close(self):
            closed.add("runtime")

        def diagnostics(self):
            return {"received_chunks": 0}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(cli, "readonly_observation", lambda settings: {})
    monkeypatch.setattr(policy_module, "create_policy", lambda settings: Engine())
    monkeypatch.setattr(robot_module, "OneroRobot", Robot)
    monkeypatch.setattr(action_module, "ActionRuntime", Runtime)
    settings = dataclasses.replace(tiny_model, processing=ProcessingConfig(inference_mode="sync"))
    if continuous:
        with pytest.raises(KeyboardInterrupt):
            cli.run(settings)
        assert sent_at == [61.0]
    else:
        with pytest.raises(RuntimeError, match="without a valid action"):
            cli.run(settings, 60)
        assert sent_at == []
    assert closed == {"runtime", "robot", "engine"}
    report_path = next((tmp_path / "runs").glob("run-*/result.json"))
    report = json.loads(report_path.read_text())
    assert report["sent_actions"] == len(sent_at)
    assert report["status"] == ("stopped_by_operator" if continuous else "failed")
    assert report["time_limit_s"] == (None if continuous else 60)


@pytest.mark.parametrize(
    "mode,combined",
    [
        ("kalman_filter", False),
        ("temporal_smoothing", False),
        ("kalman_filter", True),
    ],
)
def test_real_model_processing_preflight(tiny_model, tmp_path, monkeypatch, mode, combined):
    import lerobot_robot_onero.cli as cli

    settings = dataclasses.replace(
        tiny_model,
        processing=ProcessingConfig(inference_mode=mode, kalman_temporal_smoothing=combined),
    )
    monkeypatch.setattr(cli, "readonly_observation", lambda settings: observation())
    output = tmp_path / "processing-preview"
    result = preflight(settings, LocalPolicy(settings), output)
    assert result["ok"] and result["preview_steps"] == 2
    assert result["processing"]["inference_mode"] == mode
    assert result["processing"]["kalman_temporal_smoothing"] is combined
    assert not result["motion_attempted"] and not result["executed_actions"]
    values = np.load(output / "processed_actions.npy")
    assert values.shape == (2, 16) and np.isfinite(values).all()


def test_chunk_contract_and_structured_policy_output(tiny_model):
    engine = LocalPolicy(
        dataclasses.replace(
            tiny_model, processing=ProcessingConfig(inference_mode="temporal_smoothing")
        )
    )
    original = engine.policy.predict_action_chunk
    engine.policy.predict_action_chunk = lambda batch: {"action": original(batch)}
    assert engine.predict_chunk(observation()).shape == (2, 16)
    engine.policy.config.n_obs_steps = 2
    with pytest.raises(ValueError, match="n_obs_steps"):
        engine.validate_chunk_mode()


def test_shadow_runs_full_worker_pipeline_without_command_access(tiny_model, tmp_path, monkeypatch):
    import lerobot_robot_onero.cli as cli
    import lerobot_robot_onero.robot as robot_module

    connections = []

    class ReceiveOnlyRobot:
        def __init__(self, config):
            assert config.read_only
            self.closed = False
            connections.append(self)

        def connect(self):
            pass

        def get_observation(self):
            return observation()

        def send_action(self, action):
            pytest.fail("shadow attempted to send an action")

        def disconnect(self):
            self.closed = True

    monkeypatch.setattr(robot_module, "OneroRobot", ReceiveOnlyRobot)
    settings = dataclasses.replace(
        tiny_model,
        processing=ProcessingConfig(inference_mode="kalman_filter", kalman_temporal_smoothing=True),
    )
    result = cli.shadow(settings, 0.35, tmp_path / "shadow")
    assert result["ok"] and result["preview_actions"] > 0
    assert result["sent_actions"] == 0 and not result["executed_actions"]
    assert connections and all(c.closed for c in connections)
