import math

import pytest

from lerobot_robot_onero.protocol import (
    FEATURES,
    angle_delta,
    decode_arm,
    decode_gripper,
    encode_action,
)


def command():
    return {key: index / 100 for index, key in enumerate(FEATURES)}


def test_wire_layout_and_gripper_quantization():
    action = command()
    action.update({"left_gripper.pos": 1.2, "right_gripper.pos": 0.12345})
    arms, grips, accepted = encode_action(action)
    assert arms[:7] == [i / 100 for i in range(7)]
    assert arms[7:14] == [0] * 7
    assert arms[14:21] == [i / 100 for i in range(8, 15)]
    assert arms[21:] == [0] * 7
    assert grips == (199, 212)
    assert accepted["left_gripper.pos"] == 0.99
    assert accepted["right_gripper.pos"] == 0.12
    assert action["left_gripper.pos"] == 1.2


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.1", None])
def test_invalid_command_never_encodes(bad):
    action = command()
    action["right_joint_7.pos"] = bad
    with pytest.raises(ValueError):
        encode_action(action)


def test_exact_action_keys_required():
    action = command()
    action.pop(FEATURES[0])
    with pytest.raises(ValueError):
        encode_action(action)
    action["unknown"] = 0
    with pytest.raises(ValueError):
        encode_action(action)


def test_joint_state_uses_names_not_source_order():
    names = [f"joint{i}-l" for i in reversed(range(1, 8))]
    result = decode_arm("left", names, list(reversed(range(1, 8))))
    assert list(result.values()) == list(range(1, 8))
    with pytest.raises(ValueError):
        decode_arm("left", ["joint1-l"] * 7, [0] * 7)


def test_gripper_feedback_is_not_command_scale():
    assert decode_gripper("left", 255) == {"left_gripper.pos": 1.0}
    for bad in (-1, 256, True, 1.0):
        with pytest.raises(ValueError):
            decode_gripper("left", bad)


def test_cyclic_angle_uses_shortest_distance():
    assert angle_delta(-math.pi + 0.01, math.pi - 0.01, "left_joint_1") == pytest.approx(0.02)
    assert angle_delta(-math.pi, math.pi, "left_joint_2") == pytest.approx(-2 * math.pi)
