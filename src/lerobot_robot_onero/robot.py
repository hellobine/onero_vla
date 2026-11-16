"""Standalone LeRobot Robot for the deployed A1/H1 upper-body protocol."""

import threading
import time
from functools import partial

from lerobot.robots import Robot

from .camera import OneroRosCamera
from .config import OneroRobotConfig
from .ownership import MotionLease
from .protocol import FEATURES, decode_arm, decode_gripper, encode_action
from .ros import RosConnection


class OneroRobot(Robot):
    config_class = OneroRobotConfig
    name = "onero_a1_h1"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.cameras = {name: OneroRosCamera(c) for name, c in config.cameras.items()}
        self._ros = None
        self._publishers = {}
        self._samples = {}
        self._condition = threading.Condition()
        self._command_lock = threading.Lock()
        self._fault = None
        self._last_command = None
        self._last_observation = None
        self._stopping = threading.Event()
        self._watchdog = None
        self._connected = False
        self._lease = MotionLease(
            config.ros_domain_id, (config.joint_command_topic, config.gripper_command_topic)
        )

    @property
    def action_features(self):
        return dict.fromkeys(FEATURES, float)

    @property
    def observation_features(self):
        return {
            **self.action_features,
            **{name: (c.height, c.width, 3) for name, c in self.config.cameras.items()},
        }

    @property
    def is_connected(self):
        return self._connected and self._ros is not None and self._ros.error is None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        if self._ros is not None:
            raise RuntimeError("robot already connected or teardown incomplete")
        if not self.config.read_only:
            self._lease.acquire()
        self._fault, self._last_command, self._last_observation = None, None, None
        self._samples.clear()
        self._stopping.clear()
        try:
            self._ros = RosConnection(self.config.ros_domain_id)
            for side in ("left", "right"):
                self._ros.subscribe(
                    "joints",
                    getattr(self.config, f"{side}_joint_state_topic"),
                    partial(self._on_feedback, side, "arm"),
                )
                self._ros.subscribe(
                    "gripper",
                    getattr(self.config, f"{side}_gripper_state_topic"),
                    partial(self._on_feedback, side, "gripper"),
                )
            if not self.config.read_only:
                self._publishers = {
                    "arm_command": self._ros.publisher(
                        "arm_command", self.config.joint_command_topic
                    ),
                    "gripper_command": self._ros.publisher(
                        "gripper_command", self.config.gripper_command_topic
                    ),
                }
            self._ros.start()
            deadline = time.monotonic() + self.config.connect_timeout_s
            with self._condition:
                while len(self._samples) != 4 or any(v is None for v in self._samples.values()):
                    if self._ros.error:
                        raise RuntimeError("ROS feedback connection failed") from self._ros.error
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("robot requires four valid feedback streams")
                    self._condition.wait(min(remaining, 0.05))
            for camera in self.cameras.values():
                camera.connect()
            self._connected = True
            self.get_observation()
            if not self.config.read_only:
                self._watchdog = threading.Thread(target=self._monitor, daemon=True)
                self._watchdog.start()
        except BaseException:
            self.disconnect()
            raise

    def _on_feedback(self, side, kind, message):
        source = f"{side}_{kind}"
        try:
            values = (
                decode_arm(side, message.name, message.position)
                if kind == "arm"
                else decode_gripper(side, message.data)
            )
            sample = (values, time.monotonic())
        except (ValueError, TypeError, AttributeError):
            sample = None
        with self._condition:
            self._samples[source] = sample
            self._condition.notify_all()

    def _state(self):
        if self._ros is None or self._ros.error is not None:
            raise RuntimeError("robot ROS connection unavailable")
        with self._condition:
            now = time.monotonic()
            if len(self._samples) != 4 or any(
                s is None or now - s[1] > self.config.max_state_age_s
                for s in self._samples.values()
            ):
                raise TimeoutError("robot feedback is missing, invalid or stale")
            combined = {
                k: v for sample, _stamp in self._samples.values() for k, v in sample.items()
            }
        return {key: combined[key] for key in FEATURES}

    def get_observation(self):
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        result = self._state()
        for name, camera in self.cameras.items():
            result[name] = camera.read_latest(int(self.config.cameras[name].max_age_s * 1000))
        self._last_observation = time.monotonic()
        return result

    def _check_motion(self):
        if self.config.read_only:
            raise RuntimeError("read-only connection cannot publish commands")
        if not self.is_connected:
            raise RuntimeError("robot is not connected")
        if self._fault is not None:
            raise RuntimeError("motion fault is latched; reconnect explicitly") from self._fault
        if self._last_observation is None or (
            time.monotonic() - self._last_observation > self.config.command_timeout_s
        ):
            raise TimeoutError("observation used by the action has expired")
        self._state()
        for name, camera in self.cameras.items():
            camera.read_latest(int(self.config.cameras[name].max_age_s * 1000))

    def send_action(self, action):
        # Validate the complete command before publishing any component.
        arms, grips, accepted = encode_action(action)
        with self._command_lock:
            try:
                self._check_motion()
                # Match main's H1 absolute-position wire: forward the target as
                # produced by the policy/processing pipeline, without feedback-
                # relative step rejection or angle rebranching here.
                for kind, data in (
                    ("arm_command", arms),
                    ("gripper_command", grips[0]),
                    ("gripper_command", grips[1]),
                ):
                    self._check_motion()
                    self._ros.publish(self._publishers[kind], kind, data)
                self._last_command = time.monotonic()
            except BaseException as exc:
                self._fault = exc
                raise
        return accepted

    def _monitor(self):
        while not self._stopping.wait(min(0.05, self.config.command_timeout_s / 4)):
            with self._command_lock:
                if self._last_command is not None and (
                    time.monotonic() - self._last_command > self.config.command_timeout_s
                ):
                    self._fault = TimeoutError("command stream expired; publication disabled")
                    return

    def disconnect(self):
        self._connected = False
        self._stopping.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1)
            if self._watchdog.is_alive():
                raise RuntimeError("command watchdog teardown incomplete")
            self._watchdog = None
        errors = []
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except BaseException as exc:
                errors.append(exc)
        if self._ros is not None:
            try:
                self._ros.close()
            except BaseException as exc:
                errors.append(exc)
            else:
                self._ros = None
        if errors:
            raise BaseExceptionGroup("robot teardown incomplete; ownership retained", errors)
        self._publishers.clear()
        self._lease.release()
