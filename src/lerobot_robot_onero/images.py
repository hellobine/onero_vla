"""Explicit model-facing resize; physical camera configuration stays unchanged."""

import numpy as np


def model_image(image, resolution=None):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("model camera input must be HWC RGB uint8")
    if resolution is None or image.shape[:2] == tuple(resolution):
        return image
    height, width = resolution
    if image.shape[1] * height != image.shape[0] * width:
        raise ValueError("model-facing resize must preserve the camera aspect ratio")
    if height > image.shape[0] or width > image.shape[1]:
        raise ValueError("model-facing AREA transport must not upscale images")
    import cv2

    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
