"""Receive-only LeRobot camera, preserving source pixels and RGB order."""

import threading
import time

import numpy as np
from lerobot.cameras.camera import Camera

from .config import OneroRosCameraConfig
from .ros import RosConnection


class OneroRosCamera(Camera):
    config_class = OneroRosCameraConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._ros = None
        self._condition = threading.Condition()
        self._frame = None
        self._stamp = 0.0
        self._sequence = 0
        self._consumed = 0
        self._error = None

    @property
    def is_connected(self):
        return self._ros is not None and self._ros.error is None

    @staticmethod
    def find_cameras():
        return []

    def connect(self, warmup=True):
        if self._ros is not None:
            raise RuntimeError("camera already connected or teardown incomplete")
        import cv2

        self._cv2 = cv2
        self._frame, self._error = None, None
        self._sequence = self._consumed = 0
        self._ros = RosConnection(self.config.ros_domain_id)
        try:
            self._ros.subscribe("image", self.config.topic, self._on_image)
            self._ros.start()
            if warmup:
                self.async_read(self.config.timeout_s * 1000)
        except BaseException:
            self.disconnect()
            raise

    def _on_image(self, message):
        stamp = time.monotonic()
        try:
            bgr = self._cv2.imdecode(
                np.frombuffer(message.data, dtype=np.uint8), self._cv2.IMREAD_COLOR
            )
            if bgr is None:
                raise ValueError("invalid compressed image")
            rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
            if rgb.shape != (self.config.height, self.config.width, 3):
                raise ValueError(f"camera {self.config.topic}: unexpected image shape {rgb.shape}")
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            return
        with self._condition:
            self._frame, self._stamp, self._error = rgb, stamp, None
            self._sequence += 1
            self._condition.notify_all()

    def _check(self):
        if not self.is_connected:
            raise RuntimeError("camera ROS connection unavailable")
        if self._error is not None:
            raise RuntimeError("invalid camera frame") from self._error

    def read(self):
        return self.async_read()

    def async_read(self, timeout_ms=None):
        timeout = self.config.timeout_s if timeout_ms is None else timeout_ms / 1000
        if timeout <= 0:
            raise ValueError("camera timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._check()
                if self._frame is not None and self._sequence > self._consumed:
                    self._consumed = self._sequence
                    return self._latest(self.config.max_age_s)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"camera {self.config.topic}: no new frame")
                self._condition.wait(min(remaining, 0.05))

    def _latest(self, max_age_s):
        self._check()
        if self._frame is None or time.monotonic() - self._stamp > max_age_s:
            raise TimeoutError(f"camera {self.config.topic}: missing or stale frame")
        return self._frame.copy()

    def read_latest(self, max_age_ms=500):
        with self._condition:
            return self._latest(min(max_age_ms / 1000, self.config.max_age_s))

    def disconnect(self):
        if self._ros is not None:
            self._ros.close()
            self._ros = None
        with self._condition:
            self._frame = None
            self._condition.notify_all()
