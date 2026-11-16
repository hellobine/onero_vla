import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lerobot_robot_onero.camera import OneroRosCamera
from lerobot_robot_onero.config import OneroRosCameraConfig


def message(height=2, width=2):
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    return SimpleNamespace(data=encoded.tobytes())


@pytest.fixture
def camera():
    camera = OneroRosCamera(OneroRosCameraConfig(height=2, width=2))
    camera._cv2 = cv2
    camera._ros = SimpleNamespace(error=None, close=lambda: None)
    yield camera
    camera.disconnect()


def test_camera_decodes_rgb_without_resize_and_returns_copy(camera):
    camera._on_image(message())
    frame = camera.async_read(10)
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8
    assert list(frame[0, 0]) == [255, 0, 0]
    frame[:] = 0
    assert list(camera.read_latest()[0, 0]) == [255, 0, 0]
    with pytest.raises(TimeoutError, match="no new frame"):
        camera.async_read(5)


def test_bad_frame_never_leaves_old_frame_usable(camera):
    camera._on_image(message())
    camera._on_image(message(3, 3))
    with pytest.raises(RuntimeError, match="invalid"):
        camera.read_latest()
    camera._on_image(message())
    camera._stamp = time.monotonic() - 2
    with pytest.raises(TimeoutError, match="stale"):
        camera.read_latest()
