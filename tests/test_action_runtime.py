import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_onero.action_runtime import ActionRuntime, ChunkWorker
from lerobot_robot_onero.smoothing import ProcessingConfig


class Engine:
    def __init__(self):
        self.gate = threading.Event()
        self.started = threading.Event()
        self.failure = False

    def validate_chunk_mode(self):
        pass

    def predict_chunk(self, raw):
        self.started.set()
        self.gate.wait(1)
        if self.failure:
            raise ValueError("model failure")
        return np.zeros((16, 16))


def test_worker_allows_one_request_and_propagates_failure():
    engine = Engine()
    engine.failure = True
    worker = ChunkWorker(engine)
    try:
        assert worker.submit({}, 1, time.monotonic())
        assert engine.started.wait(1)
        assert not worker.submit({}, 2, time.monotonic())
        engine.gate.set()
        with worker.condition:
            assert worker.condition.wait_for(lambda: worker.result is not None, timeout=1)
        with pytest.raises(RuntimeError, match="worker failed"):
            worker.take()
    finally:
        engine.gate.set()
        worker.close()


def test_runtime_consumes_fresh_result_and_resets_between_instances():
    engine = Engine()
    engine.gate.set()
    settings = SimpleNamespace(
        processing=ProcessingConfig(inference_mode="kalman_filter"),
        fps=30,
        robot=SimpleNamespace(command_timeout_s=0.5),
    )
    runtime = ActionRuntime(engine, settings)
    try:
        start = time.monotonic()
        assert runtime.tick({}, 0, start) is None
        with runtime.worker.condition:
            assert runtime.worker.condition.wait_for(
                lambda: runtime.worker.result is not None, timeout=1
            )
        result = runtime.tick({}, 1, start + 1 / 30)
        assert len(result) == 16
        assert runtime.queue.latest_step == 1
    finally:
        runtime.close()
    assert not runtime.worker.thread.is_alive()
    assert not runtime.queue.actions
    with pytest.raises(RuntimeError, match="closed"):
        runtime.tick({}, 2)


def test_close_discards_late_result_and_reports_unstopped_worker():
    engine = Engine()
    worker = ChunkWorker(engine)
    assert worker.submit({}, 1, time.monotonic())
    assert engine.started.wait(1)
    with pytest.raises(TimeoutError, match="not stopped"):
        worker.close(timeout=0.01)
    engine.gate.set()
    worker.close()
    assert worker.result is None
    assert not worker.submit({}, 2, time.monotonic())
