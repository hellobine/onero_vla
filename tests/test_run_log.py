import json
from types import SimpleNamespace

import pytest

from lerobot_robot_onero.protocol import FEATURES
from lerobot_robot_onero.run_log import RunLog


def test_run_log_records_action_order_timing_and_readonly_scope(tmp_path, monkeypatch):
    import lerobot_robot_onero.run_log as module

    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    record = RunLog(SimpleNamespace(), motion=False, directory=tmp_path / "run")
    record.control_started(10.0)
    diagnostics = {"action_step": 0, "received_chunks": 1, "last_inference_s": 0.084}
    runtime = SimpleNamespace(diagnostics=lambda: dict(diagnostics))
    feedback = dict.fromkeys(FEATURES, 0.0)
    target = dict.fromkeys(FEATURES, 0.1)
    record.cycle(runtime)
    record.action(feedback, target, target)
    clock.now = 10.04
    diagnostics["action_step"] = 1
    record.cycle(runtime)
    record.action(feedback, target, target)
    clock.now = 10.1
    record.control_stopped()
    clock.now = 12.0  # Teardown time must not dilute the measured control rate.
    record.finish({"status": "completed"})
    result = json.loads((record.directory / "result.json").read_text())
    assert result["sent_actions"] == 0 and result["preview_actions"] == 2
    assert result["action_hz"] == pytest.approx(20)
    assert result["max_action_gap_ms"] == pytest.approx(40)
    assert result["mean_inference_ms"] == pytest.approx(84)
    events = [
        json.loads(line) for line in (record.directory / "events.jsonl").read_text().splitlines()
    ]
    actions = [e for e in events if e["event"] == "action"]
    assert [e["action_step"] for e in actions] == [0, 1]
    assert len(actions[0]["feedback"]) == len(actions[0]["accepted"]) == 16


def test_log_write_failure_does_not_abort_action_processing(tmp_path):
    record = RunLog(SimpleNamespace(), directory=tmp_path / "run")
    record.file.close()

    class FullDisk:
        def write(self, value):
            raise OSError("disk full")

        def close(self):
            pass

    record.file = FullDisk()
    target = dict.fromkeys(FEATURES, 0.1)
    record.command_attempt()
    record.action(target, target, target)
    record.finish({"status": "completed"})
    result = json.loads((record.directory / "result.json").read_text())
    assert result["sent_actions"] == 1
    assert result["motion_attempted"] and result["command_attempts"] == 1
    assert result["logging_error"] == "disk full"
