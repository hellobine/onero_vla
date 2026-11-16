import json
import math
from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_onero.protocol import FEATURES
from lerobot_robot_onero.smoothing import (
    ActionFilter,
    ActionQueue,
    AdaptiveKalmanFilter,
    CircularAdaptiveKalmanFilter,
    ProcessingConfig,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("inference_mode", "typo"),
        ("kalman_temporal_smoothing", "false"),
        ("kalman_process_variance", 0),
        ("kalman_measurement_variance", math.nan),
        ("kalman_threshold", True),
        ("kalman_scale_factor", -1),
        ("min_smooth_steps", 0),
        ("min_smooth_steps", 1.5),
        ("chunk_size_threshold", 0),
        ("chunk_size_threshold", math.inf),
        ("max_chunk_age_s", -1),
        ("aggregate_fn_name", "typo"),
    ],
)
def test_invalid_processing_parameters(field, value):
    with pytest.raises(ValueError):
        ProcessingConfig(**{field: value})


def test_combination_must_be_explicit_kalman_mode():
    with pytest.raises(ValueError):
        ProcessingConfig(kalman_temporal_smoothing=True)
    assert ProcessingConfig(inference_mode="kalman_filter", kalman_temporal_smoothing=True).temporal


def test_kalman_matches_frozen_main_reference():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/smoothing_reference.json").read_text()
    )
    assert reference["source_commit"].startswith("4fcfbcf2")
    for case in reference["kalman"]:
        cls = CircularAdaptiveKalmanFilter if case["circular"] else AdaptiveKalmanFilter
        filter_ = cls(delta_t=1 / case["fps"])
        np.testing.assert_allclose(
            [filter_.step(v) for v in case["inputs"]], case["outputs"], rtol=1e-12, atol=1e-12
        )


def test_all_dimensions_filtered_once_and_session_reset():
    config = ProcessingConfig(inference_mode="kalman_filter")
    filters = ActionFilter(config, 30)
    first = dict.fromkeys(FEATURES, 0.0)
    filters.apply(first)
    target = dict.fromkeys(FEATURES, 1.0)
    actual = filters.apply(target)
    reference = AdaptiveKalmanFilter(delta_t=1 / 30)
    reference.step(0)
    expected = reference.step(1)
    assert all(v == pytest.approx(expected) for v in actual.values())
    filters.reset()
    assert filters.apply(target) == target


def test_invalid_action_does_not_partially_advance_filters():
    filters = ActionFilter(ProcessingConfig(inference_mode="kalman_filter"), 30)
    action = dict.fromkeys(FEATURES, 0.2)
    action[FEATURES[-1]] = math.nan
    with pytest.raises(ValueError):
        filters.apply(action)
    assert not any(f.initialized for f in filters.filters.values())


@pytest.mark.parametrize("index", [0, 8])
def test_circular_kalman_keeps_boundary_near_pi(index):
    filter_ = ActionFilter(ProcessingConfig(inference_mode="kalman_filter"), 30)
    action = dict.fromkeys(FEATURES, 0.0)
    action[FEATURES[index]] = math.pi - 0.02
    filter_.apply(action)
    action[FEATURES[index]] = -math.pi + 0.02
    result = filter_.apply(action)
    assert abs(result[FEATURES[index]]) > 3


def test_temporal_minimum_width_tail_padding_matches_reference():
    config = ProcessingConfig(inference_mode="temporal_smoothing", min_smooth_steps=4)
    queue = ActionQueue(config)
    old = np.zeros((2, 16))
    old[:, 1] = [0.0, 2.0]
    new = np.zeros((4, 16))
    new[:, 1] = 10.0
    queue.accept(0, old, 0.0)
    queue.accept(0, new, 0.1)
    expected = [0.0, 2 * 2 / 3 + 10 / 3, 2 / 3 + 20 / 3, 10.0]
    assert [a.values[1] for a in queue.actions] == pytest.approx(expected)
    assert [a.captured_at for a in queue.actions] == [0.0, 0.0, 0.0, 0.1]


def test_temporal_circular_overlap_never_blends_through_zero():
    queue = ActionQueue(ProcessingConfig(inference_mode="temporal_smoothing", min_smooth_steps=3))
    old, new = np.zeros((3, 16)), np.zeros((3, 16))
    old[:, [0, 8]] = math.pi - 0.02
    new[:, [0, 8]] = -math.pi + 0.02
    queue.accept(0, old, 0.0)
    queue.accept(0, new, 0.1)
    assert all(abs(a.values[0]) > 3 and abs(a.values[8]) > 3 for a in queue.actions)


def test_queue_drops_expired_steps_and_never_replays():
    queue = ActionQueue(ProcessingConfig())
    chunk = np.zeros((5, 16))
    chunk[:, 1] = range(5)
    queue.advance(1)
    queue.accept(0, chunk, 0.0)
    assert [a.step for a in queue.actions] == [2, 3, 4]
    assert queue.pop(2, 0.2)[FEATURES[1]] == 2
    with pytest.raises(ValueError, match="once"):
        queue.pop(2, 0.3)
    assert not queue.accept(0, chunk[:2], 0.2)
    assert queue.actions == []


def test_blending_preserves_oldest_contributor_expiry():
    queue = ActionQueue(ProcessingConfig(inference_mode="temporal_smoothing", max_chunk_age_s=0.5))
    queue.accept(0, np.zeros((3, 16)), 0.0)
    queue.accept(0, np.ones((3, 16)), 0.4)
    with pytest.raises(TimeoutError, match="expired"):
        queue.pop(0, 0.6)


def test_take_latest_mode_does_not_apply_temporal_blending():
    queue = ActionQueue(
        ProcessingConfig(inference_mode="kalman_filter", aggregate_fn_name="latest_only")
    )
    queue.accept(0, np.zeros((3, 16)), 0.0)
    queue.accept(0, np.ones((3, 16)), 0.1)
    assert all(np.array_equal(a.values, np.ones(16)) for a in queue.actions)


@pytest.mark.parametrize(
    "name,weight", [("weighted_average", 0.3), ("average", 0.5), ("conservative", 0.7)]
)
def test_normal_overlap_uses_upstream_same_timestep_aggregation(name, weight):
    queue = ActionQueue(ProcessingConfig(inference_mode="kalman_filter", aggregate_fn_name=name))
    old, new = np.zeros((3, 16)), np.zeros((3, 16))
    old[:, 1] = 2
    new[:, 1] = 10
    old[:, [0, 8]] = math.pi - 0.02
    new[:, [0, 8]] = -math.pi + 0.02
    queue.accept(0, old, 0.0)
    queue.accept(1, new, 0.1)
    assert queue.actions[0].values[1] == pytest.approx(weight * 2 + (1 - weight) * 10)
    assert queue.actions[0].values[0] == pytest.approx(
        weight * (math.pi - 0.02) + (1 - weight) * (-math.pi + 0.02)
    )
    assert queue.actions[0].captured_at == 0.0
    assert queue.actions[-1].captured_at == 0.1


def test_default_aggregation_matches_original_main_client_reference():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/action_queue_main_reference.json").read_text()
    )
    assert reference["reference_main"].startswith("4fcfbcf2")
    for case in reference["cases"]:
        queue = ActionQueue(ProcessingConfig(inference_mode="kalman_filter"))
        if case["latest"] >= 0:
            queue.advance(case["latest"])
        if case["old"]:
            queue.accept(case["old_start"], case["old"], 0.0)
        queue.accept(case["new_start"], case["new"], 0.1)
        assert [a.step for a in queue.actions] == case["expected_steps"]
        if queue.actions:
            np.testing.assert_allclose(
                [a.values for a in queue.actions], case["expected_actions"], rtol=1e-6, atol=1e-6
            )
