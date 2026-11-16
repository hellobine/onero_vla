"""Output Kalman filters and chunk-overlap blending, independent of the platform.

Scalar Kalman equations/defaults follow the reference main commit 4fcfbcf2.
Circular blending extends its linear overlap rule for A1/H1 joint 1.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import positive
from .protocol import CYCLIC, FEATURES, finite, validate_action

CYCLIC_INDICES = tuple(i for i, key in enumerate(FEATURES) if key.removesuffix(".pos") in CYCLIC)


@dataclass(frozen=True)
class ProcessingConfig:
    inference_mode: str = "sync"
    kalman_temporal_smoothing: bool = False
    kalman_process_variance: float = 0.5
    kalman_measurement_variance: float = 5.0
    kalman_threshold: float = 0.075
    kalman_scale_factor: float = 15.0
    min_smooth_steps: int = 8
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    max_chunk_age_s: float = 5.0

    def __post_init__(self):
        if self.inference_mode not in ("sync", "kalman_filter", "temporal_smoothing"):
            raise ValueError("inference_mode must be sync, kalman_filter or temporal_smoothing")
        if type(self.kalman_temporal_smoothing) is not bool:
            raise ValueError("kalman_temporal_smoothing must be boolean")
        if self.kalman_temporal_smoothing and self.inference_mode != "kalman_filter":
            raise ValueError("kalman_temporal_smoothing requires kalman_filter mode")
        for name in (
            "kalman_process_variance",
            "kalman_measurement_variance",
            "kalman_threshold",
            "kalman_scale_factor",
            "max_chunk_age_s",
        ):
            positive(getattr(self, name), name)
        if type(self.min_smooth_steps) is not int or not 1 <= self.min_smooth_steps <= 100000:
            raise ValueError("min_smooth_steps must be an integer in [1, 100000]")
        if not 0 < finite(self.chunk_size_threshold, "chunk_size_threshold") <= 1:
            raise ValueError("chunk_size_threshold must be in (0, 1]")
        if self.aggregate_fn_name not in (
            "weighted_average",
            "latest_only",
            "average",
            "conservative",
        ):
            raise ValueError("unsupported aggregate_fn_name")

    @property
    def temporal(self):
        return self.inference_mode == "temporal_smoothing" or self.kalman_temporal_smoothing

    @property
    def chunked(self):
        return self.inference_mode != "sync"


class AdaptiveKalmanFilter:
    def __init__(
        self,
        process_variance=0.5,
        measurement_variance=5.0,
        threshold=0.075,
        scale_factor=15.0,
        delta_t=1 / 30,
    ):
        for name, value in locals().copy().items():
            if name != "self":
                positive(value, name)
        self.A = np.array([[1.0, delta_t], [0.0, 1.0]])
        self.Q = process_variance * np.eye(2)
        self.R = measurement_variance * np.eye(1)
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.prev_measurement = np.zeros(1)
        self.initialized = False

    def initialize_state_from_measurement(self, measurement):
        self.x = np.array([finite(measurement[0], "measurement"), 0.0])
        self.P = 0.1 * np.eye(2)
        self.prev_measurement = np.array([self.x[0]])
        self.initialized = True

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        measurement = np.asarray([finite(measurement[0], "measurement")])
        delta = np.abs(measurement - self.prev_measurement)
        # The reference deliberately replaces the initial R with 7.5 or scale_factor.
        self.R = (self.scale_factor if np.any(delta > self.threshold) else 7.5) * np.eye(1)
        h = np.array([[1.0, 0.0]])
        gain = self.P @ h.T @ np.linalg.inv(h @ self.P @ h.T + self.R)
        self.x += gain @ (measurement - h @ self.x)
        self.P = (np.eye(2) - gain @ h) @ self.P
        self.prev_measurement = measurement
        return self.x

    def step(self, value):
        measurement = np.array([finite(value, "measurement")])
        if not self.initialized:
            self.initialize_state_from_measurement(measurement)
        self.predict()
        return float(self.update(measurement)[0])


class CircularAdaptiveKalmanFilter(AdaptiveKalmanFilter):
    def update(self, measurement):
        delta = (measurement[0] - self.prev_measurement[0] + np.pi) % (2 * np.pi) - np.pi
        continuous = np.array([self.prev_measurement[0] + delta])
        result = super().update(continuous).copy()
        result[0] = (result[0] + np.pi) % (2 * np.pi) - np.pi
        return result


class ActionFilter:
    """Owned by the dispatch thread; apply once per consumed action, then validate on the robot."""

    def __init__(self, config, fps):
        self.config, self.fps = config, positive(fps, "fps")
        self.reset()

    def reset(self):
        self.filters = {
            key: (CircularAdaptiveKalmanFilter if i in CYCLIC_INDICES else AdaptiveKalmanFilter)(
                process_variance=self.config.kalman_process_variance,
                measurement_variance=self.config.kalman_measurement_variance,
                threshold=self.config.kalman_threshold,
                scale_factor=self.config.kalman_scale_factor,
                delta_t=1 / self.fps,
            )
            for i, key in enumerate(FEATURES)
        }

    def apply(self, action):
        values = validate_action(action)
        if self.config.inference_mode != "kalman_filter":
            return values
        return {key: self.filters[key].step(values[key]) for key in FEATURES}


@dataclass(frozen=True)
class QueuedAction:
    step: int
    values: np.ndarray
    captured_at: float


def blend_actions(previous, incoming, old_weight, *, circular=True):
    value = old_weight * previous.values + (1 - old_weight) * incoming.values
    for index in CYCLIC_INDICES if circular else ():
        delta = (incoming.values[index] - previous.values[index] + math.pi) % (
            2 * math.pi
        ) - math.pi
        value[index] = (previous.values[index] + (1 - old_weight) * delta + math.pi) % (
            2 * math.pi
        ) - math.pi
    capture = (
        min(previous.captured_at, incoming.captured_at) if old_weight > 0 else incoming.captured_at
    )
    return QueuedAction(incoming.step, value, capture)


class ActionQueue:
    """Single-consumer queue. Worker results are merged only on the dispatch thread."""

    def __init__(self, config):
        self.config = config
        self.latest_step = -1
        self.actions = []
        self.last_chunk_size = 0

    def advance(self, step):
        if step < self.latest_step:
            raise ValueError("action cursor cannot move backwards")
        self.latest_step = step
        self.actions = [a for a in self.actions if a.step > step]

    def accept(self, start_step, actions, captured_at):
        if type(start_step) is not int or start_step < 0:
            raise ValueError("chunk start_step must be a nonnegative integer")
        captured_at = finite(captured_at, "captured_at")
        values = np.array(actions, dtype=np.float64, copy=True)
        if (
            values.ndim != 2
            or values.shape[1] != 16
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("action chunk must be a finite nonempty [T,16] array")
        new = [
            QueuedAction(start_step + i, row.copy(), captured_at)
            for i, row in enumerate(values)
            if start_step + i > self.latest_step
        ]
        if not new:
            self.actions = []
            return False
        old = [a for a in self.actions if a.step > self.latest_step]
        self.last_chunk_size = len(values)
        if self.config.temporal and old:
            width = min(max(len(old), self.config.min_smooth_steps), len(new))
            blended = []
            for i in range(width):
                previous, incoming = old[min(i, len(old) - 1)], new[i]
                weight = 1 - i / max(width - 1, 1)
                blended.append(blend_actions(previous, incoming, weight))
            new = blended + new[width:]
        elif old:
            old_weight = {
                "weighted_average": 0.3,
                "latest_only": 0.0,
                "average": 0.5,
                "conservative": 0.7,
            }[self.config.aggregate_fn_name]
            previous = {a.step: a for a in old}
            new = [
                blend_actions(previous[a.step], a, old_weight, circular=False)
                if a.step in previous and old_weight
                else a
                for a in new
            ]
        self.actions = new
        return True

    def pop(self, step, now):
        now = finite(now, "now")
        if step <= self.latest_step:
            raise ValueError("a timestep may be consumed only once")
        self.advance(step - 1)
        item = next((a for a in self.actions if a.step == step), None)
        self.advance(step)
        if item is None:
            return None
        if now < item.captured_at:
            raise ValueError("queued action capture time is in the future")
        if now - item.captured_at > self.config.max_chunk_age_s:
            raise TimeoutError("queued action provenance expired")
        return {key: float(value) for key, value in zip(FEATURES, item.values, strict=True)}

    def needs_prediction(self):
        return (
            not self.actions
            or len(self.actions) <= self.last_chunk_size * self.config.chunk_size_threshold
        )
