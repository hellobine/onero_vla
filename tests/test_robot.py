import time
from types import SimpleNamespace

import numpy as np
import pytest

import lerobot_robot_onero.robot as robot_module
from lerobot_robot_onero.config import OneroRobotConfig, OneroRosCameraConfig
from lerobot_robot_onero.ownership import MotionLease
from lerobot_robot_onero.protocol import FEATURES
from lerobot_robot_onero.robot import OneroRobot


class FakeRos:
    instances = []

    def __init__(self, domain):
        self.domain = domain
        self.error = None
        self.subscribers = []
        self.publishers = []
        self.sent = []
        self.fail_on = None
        self.close_error = False
        self.__class__.instances.append(self)

    def subscribe(self, kind, topic, callback):
        self.subscribers.append((kind, topic, callback))

    def publisher(self, kind, topic):
        self.publishers.append((kind, topic))
        return kind

    def publish(self, publisher, kind, data):
        if self.fail_on == len(self.sent) + 1:
            raise RuntimeError("publisher failed")
        self.sent.append((kind, data))

    def start(self):
        for kind, topic, callback in self.subscribers:
            side = "r" if "right" in topic else "l"
            if kind == "joints":
                callback(
                    SimpleNamespace(
                        name=[f"joint{i}-{side}" for i in range(1, 8)], position=[0.0] * 7
                    )
                )
            elif kind == "gripper":
                callback(SimpleNamespace(data=128))

    def close(self):
        if self.close_error:
            raise RuntimeError("teardown failed")


class FakeCamera:
    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.stale = False

    def connect(self):
        self.is_connected = True

    def read_latest(self, _age):
        if self.stale:
            raise TimeoutError("stale camera")
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def disconnect(self):
        self.is_connected = False


@pytest.fixture
def make_robot(monkeypatch, tmp_path):
    FakeRos.instances = []
    monkeypatch.setattr(robot_module, "RosConnection", FakeRos)
    monkeypatch.setattr(robot_module, "OneroRosCamera", FakeCamera)
    robots = []

    def make(read_only=True):
        cameras = {
            n: OneroRosCameraConfig(topic=f"/{n}/camera/rgb", height=32, width=32)
            for n in ("head", "left", "right")
        }
        robot = OneroRobot(
            OneroRobotConfig(
                id="test", calibration_dir=tmp_path, read_only=read_only, cameras=cameras
            )
        )
        robots.append(robot)
        return robot

    yield make
    for robot in robots:
        if robot._ros:
            robot._ros.close_error = False
        robot.disconnect()


def action():
    return dict.fromkeys(FEATURES, 0.02)


def test_readonly_never_creates_publishers(make_robot):
    robot = make_robot()
    robot.connect()
    observation = robot.get_observation()
    assert list(observation) == [*FEATURES, "head", "left", "right"]
    assert robot._ros.publishers == []
    with pytest.raises(RuntimeError, match="read-only"):
        robot.send_action(action())
    assert robot._ros.sent == []


def test_motion_requires_lease_and_publishes_complete_wire(make_robot):
    first, second = make_robot(False), make_robot(False)
    first.connect()
    with pytest.raises(BlockingIOError):
        second.connect()
    first.send_action(action())
    assert [kind for kind, _data in first._ros.sent] == [
        "arm_command",
        "gripper_command",
        "gripper_command",
    ]
    assert first._ros.sent[0][1][:7] == pytest.approx([0.02] * 7)
    first.disconnect()
    second.connect()


def test_partially_overlapping_command_channels_cannot_both_be_owned():
    first = MotionLease(222, ("/arms-test", "/gripper-one"))
    second = MotionLease(222, ("/arms-test", "/gripper-two"))
    try:
        first.acquire()
        with pytest.raises(BlockingIOError):
            second.acquire()
        assert not second.fds
        first.release()
        second.acquire()
    finally:
        first.release()
        second.release()


@pytest.mark.parametrize("fault", ["state", "camera", "observation"])
def test_invalid_or_stale_inputs_block_all_publications(make_robot, fault):
    robot = make_robot(False)
    robot.connect()
    target = action()
    if fault == "state":
        data, _ = robot._samples["left_arm"]
        robot._samples["left_arm"] = (data, time.monotonic() - 2)
    elif fault == "camera":
        robot.cameras["head"].stale = True
    elif fault == "observation":
        robot._last_observation = time.monotonic() - 2
    with pytest.raises((TimeoutError, ValueError)):
        robot.send_action(target)
    assert robot._ros.sent == []
    assert robot._fault is not None


def test_absolute_targets_are_forwarded_without_delta_limit_or_rebranch(make_robot):
    robot = make_robot(False)
    robot.connect()
    for side, first in (("left", 3.13), ("right", -3.13)):
        values, _ = robot._samples[f"{side}_arm"]
        values[f"{side}_joint_1.pos"] = first
    values, _ = robot._samples["right_arm"]
    values["right_joint_6.pos"] = 1.0584042072296143
    robot.get_observation()
    target = action()
    target.update(
        {
            "left_joint_1.pos": -3.13,
            "right_joint_1.pos": 3.13,
            "left_joint_2.pos": 2.0,
            "right_joint_6.pos": 0.850604772567749,
        }
    )
    accepted = robot.send_action(target)
    wire = robot._ros.sent[0][1]
    for offset, side in ((0, "left"), (14, "right")):
        for index in range(7):
            key = f"{side}_joint_{index + 1}.pos"
            assert wire[offset + index] == target[key]
            assert accepted[key] == target[key]
    assert wire[7:14] == wire[21:] == [0.0] * 7
    assert robot._state()["right_joint_6.pos"] == 1.0584042072296143
    assert [kind for kind, _ in robot._ros.sent] == [
        "arm_command",
        "gripper_command",
        "gripper_command",
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_or_boolean_targets_are_rejected_before_publication(make_robot, value):
    robot = make_robot(False)
    robot.connect()
    target = action()
    target["right_joint_6.pos"] = value
    with pytest.raises(ValueError, match="finite"):
        robot.send_action(target)
    assert robot._ros.sent == []


def test_partial_publish_failure_latches_and_does_not_retry(make_robot):
    robot = make_robot(False)
    robot.connect()
    robot._ros.fail_on = 2
    with pytest.raises(RuntimeError, match="publisher failed"):
        robot.send_action(action())
    assert len(robot._ros.sent) == 1
    with pytest.raises(RuntimeError, match="latched"):
        robot.send_action(action())
    assert len(robot._ros.sent) == 1


def test_failed_teardown_retains_control_ownership(make_robot):
    first, second = make_robot(False), make_robot(False)
    first.connect()
    first._ros.close_error = True
    with pytest.raises(ExceptionGroup):
        first.disconnect()
    with pytest.raises(BlockingIOError):
        second.connect()
    first._ros.close_error = False
    first.disconnect()
    second.connect()


def test_expiring_stream_latches_without_publishing_a_stop(make_robot):
    robot = make_robot(False)
    robot.connect()
    robot.send_action(action())
    robot._last_command = time.monotonic() - 2
    deadline = time.monotonic() + 1
    while robot._fault is None and time.monotonic() < deadline:
        robot._stopping.wait(0.01)
    assert isinstance(robot._fault, TimeoutError)
    assert len(robot._ros.sent) == 3
    with pytest.raises(RuntimeError, match="latched"):
        robot.send_action(action())
