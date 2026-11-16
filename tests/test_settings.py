from pathlib import Path

import pytest

from lerobot_robot_onero.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_customer_templates_are_valid():
    for name in ("a1", "h1"):
        settings = load_settings(ROOT / f"configs/{name}.example.yaml")
        assert settings.robot.model == name
        assert settings.robot.read_only is True


@pytest.mark.parametrize(
    "text",
    [
        "model: {path: /model}\nmodel: {path: /other}",
        "model: {path: /model}\nrobot: {read_only: false}",
        "model: {path: /model}\nruntime: {fps: .nan}",
        "model: {path: /model}\nunknown: true",
        "model: {path: /model}\nros: {rmw_implementation: rmw_zenoh_cpp}",
        "model: {path: /model}\nrobot: {max_joint_step_rad: 0.15}",
        "model: {path: /model}\nruntime: {fps: 1}",
        "model: {path: /model}\nruntime: {kalman_temporal_smoothing: true}",
        "model: {path: /model}\nruntime: {inference_mode: kalman_filter, kalman_threshold: .inf}",
        "model: {path: /model}\n"
        "runtime: {inference_mode: temporal_smoothing, min_smooth_steps: false}",
    ],
)
def test_reject_bad_configuration_before_connect(tmp_path, text):
    path = tmp_path / "test.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_settings(path)


def test_relative_model_path_uses_config_directory(tmp_path):
    path = tmp_path / "station.yaml"
    path.write_text("model: {path: ../weights, device: cpu}")
    assert load_settings(path).model_path == (tmp_path.parent / "weights").resolve()


def test_processing_cli_overrides_are_validated_before_hardware(tmp_path, capsys):
    from lerobot_robot_onero.cli import main

    path = tmp_path / "station.yaml"
    path.write_text("model: {path: /weights}")
    assert (
        main(
            [
                "config",
                "--config",
                str(path),
                "--inference-mode",
                "kalman_filter",
                "--kalman-temporal-smoothing",
            ]
        )
        == 0
    )
    assert '"kalman_temporal_smoothing": true' in capsys.readouterr().out
    assert main(["config", "--config", str(path), "--kalman-temporal-smoothing"]) == 1
