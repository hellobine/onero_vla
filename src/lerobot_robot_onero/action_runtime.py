"""Local asynchronous chunk worker and dispatch-side processing; never owns a robot."""

import threading
import time

import numpy as np

from .protocol import FEATURES
from .smoothing import ActionFilter, ActionQueue


class ChunkWorker:
    def __init__(self, engine):
        self.engine = engine
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.busy = False
        self.stopping = False
        self.thread = threading.Thread(target=self._loop, name="onero-chunk-policy", daemon=True)
        self.thread.start()

    def submit(self, raw, start_step, captured_at):
        with self.condition:
            if self.stopping or self.busy:
                return False
            self.busy = True
            self.pending = (raw, start_step, captured_at)
            self.condition.notify_all()
            return True

    def _loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.pending is not None)
                if self.stopping:
                    return
                raw, start, stamp = self.pending
                self.pending = None
            try:
                value = (start, self.engine.predict_chunk(raw), stamp)
            except BaseException as exc:
                value = exc
            with self.condition:
                if self.stopping:
                    return
                self.result = value
                self.condition.notify_all()

    def take(self):
        with self.condition:
            value = self.result
            if value is not None:
                self.result = None
                self.busy = False
        if isinstance(value, BaseException):
            raise RuntimeError("chunk policy worker failed") from value
        return value

    def close(self, timeout=2.0):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("policy worker has not stopped; motion must remain disabled")


class ActionRuntime:
    """Call once per wall-clock control step, after obtaining fresh physical feedback."""

    def __init__(self, engine, settings):
        self.engine, self.settings = engine, settings
        self.queue = ActionQueue(settings.processing)
        self.filter = ActionFilter(settings.processing, settings.fps)
        self.worker = None
        self.closed = False
        if settings.processing.chunked:
            engine.validate_chunk_mode()
            self.worker = ChunkWorker(engine)

    def tick(self, raw, step, now=None):
        if self.closed:
            raise RuntimeError("action runtime is closed")
        now = time.monotonic() if now is None else now
        if self.worker is None:
            start = time.monotonic()
            action = self.engine.select(raw)
            if time.monotonic() - start > self.settings.robot.command_timeout_s:
                raise TimeoutError("policy action arrived after its command deadline")
            return self.filter.apply(action)
        self.queue.advance(step - 1)
        result = self.worker.take()
        if result is not None:
            start, actions, captured_at = result
            self.queue.accept(start, actions, captured_at)
        action = self.queue.pop(step, now)
        # The observation at step N predicts N+1 onwards, as in the reference wire.
        # One outstanding request bounds memory and prevents an inference backlog.
        if self.queue.needs_prediction():
            self.worker.submit(raw, step + 1, now)
        return None if action is None else self.filter.apply(action)

    def close(self):
        self.closed = True
        self.queue.actions.clear()
        if self.worker is not None:
            self.worker.close()


def preview_processing(engine, raw, settings):
    """Deterministic no-motion preview of the same queue/filter used for dispatch."""
    config = settings.processing
    filter_ = ActionFilter(config, settings.fps)
    if not config.chunked:
        action = engine.select(raw)
        values = np.array([[action[k] for k in FEATURES]])
        processed = filter_.apply(action)
        return values, np.array([[processed[k] for k in FEATURES]])
    engine.validate_chunk_mode()
    first = engine.predict_chunk(raw)
    second = engine.predict_chunk(raw)
    queue = ActionQueue(config)
    queue.accept(0, first, 0.0)
    output = [filter_.apply(queue.pop(0, 0.0))]
    queue.accept(1, second, 0.0)
    for step in range(1, len(second) + 1):
        value = queue.pop(step, step / settings.fps)
        if value is None:
            raise ValueError("processing preview unexpectedly lost an action")
        output.append(filter_.apply(value))
    return np.concatenate([first[:1], second]), np.array([[v[k] for k in FEATURES] for v in output])
