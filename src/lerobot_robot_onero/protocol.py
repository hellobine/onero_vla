"""A1/H1 upper-body wire contract; radians and normalized gripper commands."""

import math
from collections.abc import Mapping, Sequence

LEFT = tuple(f"left_joint_{i}" for i in range(1, 8))
RIGHT = tuple(f"right_joint_{i}" for i in range(1, 8))
JOINTS = (*LEFT, "left_gripper", *RIGHT, "right_gripper")
FEATURES = tuple(f"{name}.pos" for name in JOINTS)
CYCLIC = frozenset((LEFT[0], RIGHT[0]))


def finite(value: object, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def validate_action(action: Mapping[str, object]) -> dict[str, float]:
    if set(action) != set(FEATURES):
        raise ValueError("action must contain exactly the ordered 16 position features")
    return {name: finite(action[name], name) for name in FEATURES}


def encode_action(action: Mapping[str, object]) -> tuple[list[float], tuple[int, int], dict]:
    values = validate_action(action)
    arms = [0.0] * 28
    for offset, names in ((0, LEFT), (14, RIGHT)):
        for index, name in enumerate(names):
            arms[offset + index] = values[f"{name}.pos"]
    codes = []
    for side, offset in (("left", 100), ("right", 200)):
        key = f"{side}_gripper.pos"
        code = int(min(0.99, max(0.0, values[key])) * 100.0 + offset)
        codes.append(code)
        values[key] = (code - offset) / 100.0
    return arms, tuple(codes), values


def decode_arm(side: str, names: Sequence[str], positions: Sequence[object]) -> dict[str, float]:
    if side not in ("left", "right") or len(names) != len(positions):
        raise ValueError("invalid joint feedback layout")
    if len(set(names)) != len(names):
        raise ValueError("duplicate joint feedback names")
    wire = dict(zip(names, positions, strict=True))
    suffix = "l" if side == "left" else "r"
    required = [f"joint{i}-{suffix}" for i in range(1, 8)]
    if not set(required).issubset(wire):
        raise ValueError(f"{side} feedback lacks required named joints")
    return {
        f"{side}_joint_{i}.pos": finite(wire[name], name)
        for i, name in enumerate(required, start=1)
    }


def decode_gripper(side: str, value: object) -> dict[str, float]:
    if side not in ("left", "right") or type(value) is not int or not 0 <= value <= 255:
        raise ValueError("invalid gripper feedback")
    return {f"{side}_gripper.pos": value / 255.0}


def angle_delta(target: float, present: float, joint: str) -> float:
    delta = target - present
    return (delta + math.pi) % (2 * math.pi) - math.pi if joint in CYCLIC else delta
