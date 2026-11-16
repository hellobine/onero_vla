"""Buffered operator diagnostics; never controls or modifies model/robot values."""

import dataclasses
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .protocol import FEATURES
from .release import LEROBOT_COMMIT, LEROBOT_VERSION, VERSION


class RunLog:
    def __init__(self, settings, *, motion=True, directory=None):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(directory or f"runs/run-{stamp}-{uuid.uuid4().hex[:8]}")
        self.motion = motion
        self.file = None
        self.io_error = None
        self.actions = 0
        self.command_attempts = 0
        self.epoch = None
        self.stopped_at = None
        self.last_action_time = None
        self.max_action_gap = 0.0
        self.last_chunks = 0
        self.inference_sum = 0.0
        self.inference_max = 0.0
        self.last_diagnostics = {}
        self.last_flush = time.monotonic()
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.file = (self.directory / "events.jsonl").open("w", buffering=65536)
            if dataclasses.is_dataclass(settings):
                effective = dataclasses.asdict(settings)
                effective["robot"]["read_only"] = not motion
                self._document("settings.json", effective)
            print(f"Run log: {self.directory}", file=sys.stderr)
        except OSError as exc:
            self._failed(exc)
        self.event("start", motion=motion)

    def _failed(self, exc):
        if self.io_error is None:
            print(f"Run log unavailable: {exc}", file=sys.stderr)
        self.io_error = str(exc)

    def _document(self, name, value):
        try:
            (self.directory / name).write_text(
                json.dumps(value, ensure_ascii=False, allow_nan=False, default=str, indent=2) + "\n"
            )
        except OSError as exc:
            self._failed(exc)

    def event(self, kind, **values):
        if self.file is None or self.io_error is not None:
            return
        now = time.monotonic()
        try:
            self.file.write(json.dumps({"event": kind, "monotonic_s": now, **values}) + "\n")
            if now - self.last_flush >= 1:
                self.file.flush()
                self.last_flush = now
        except OSError as exc:
            self._failed(exc)

    def model(self, engine):
        self._document("model.json", getattr(engine, "identity", {}))

    def control_started(self, epoch):
        self.epoch = epoch
        self.event("control_started")

    def cycle(self, runtime):
        self.last_diagnostics = runtime.diagnostics()
        count = self.last_diagnostics["received_chunks"]
        if count != self.last_chunks:
            elapsed = self.last_diagnostics["last_inference_s"]
            self.inference_sum += elapsed
            self.inference_max = max(self.inference_max, elapsed)
            self.last_chunks = count
            self.event("chunk", **self.last_diagnostics)

    def control_stopped(self):
        self.stopped_at = time.monotonic()

    def action(self, feedback, requested, accepted):
        now = time.monotonic()
        if self.last_action_time is not None:
            self.max_action_gap = max(self.max_action_gap, now - self.last_action_time)
        self.last_action_time = now
        self.actions += 1
        self.event(
            "action",
            index=self.actions - 1,
            feedback=[feedback.get(k) for k in FEATURES],
            requested=[float(requested[k]) for k in FEATURES],
            accepted=[float(accepted[k]) for k in FEATURES],
            **self.last_diagnostics,
        )

    def command_attempt(self):
        self.command_attempts += 1

    def finish(self, outcome):
        stopped = self.stopped_at if self.stopped_at is not None else time.monotonic()
        elapsed = stopped - self.epoch if self.epoch is not None else 0.0
        result = {
            "adapter_version": VERSION,
            "lerobot_version": LEROBOT_VERSION,
            "lerobot_commit": LEROBOT_COMMIT,
            **outcome,
            "motion_enabled": self.motion,
            "motion_attempted": bool(self.command_attempts),
            "command_attempts": self.command_attempts,
            "sent_actions": self.actions if self.motion else 0,
            "preview_actions": 0 if self.motion else self.actions,
            "control_elapsed_s": elapsed,
            "action_hz": self.actions / elapsed if elapsed else None,
            "max_action_gap_ms": self.max_action_gap * 1000,
            "mean_inference_ms": self.inference_sum / self.last_chunks * 1000
            if self.last_chunks
            else None,
            "max_inference_ms": self.inference_max * 1000 if self.last_chunks else None,
            "diagnostics": self.last_diagnostics,
            "logging_error": self.io_error,
        }
        self.event("finished", **result)
        if self.file is not None:
            try:
                self.file.close()
            except OSError as exc:
                self._failed(exc)
        result["logging_error"] = self.io_error
        self._document("result.json", result)
