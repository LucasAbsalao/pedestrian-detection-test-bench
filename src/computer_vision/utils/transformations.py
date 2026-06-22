from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import cv2


def fog(image: NDArray, depth_map: NDArray, minimum_distance: float,
        airlight: float | None = None, per_channel_airlight: bool = False) -> NDArray:
    image = image.astype(np.float32)

    k = 3 / minimum_distance

    if airlight is None:
        if per_channel_airlight and image.ndim == 3:
            atmospheric_light = np.mean(image, axis=(0, 1))
        else:
            atmospheric_light = np.mean(image)
    else:
        atmospheric_light = airlight

    if depth_map.ndim < 3 and image.ndim == 3:
        channels = image.shape[2]
        beer_lambert = np.repeat(np.exp((-k) * depth_map)[..., np.newaxis], channels, axis=2)
    else:
        beer_lambert = depth_map

    new_image = image * beer_lambert + atmospheric_light * (1 - beer_lambert)

    return new_image

def gaussian_noise(image: NDArray, mean: float = 0.0, stdev: float = 0.2) -> NDArray:
    noise = np.random.normal(loc=mean, scale=stdev, size=image.shape)
    new_image = image + noise

    return np.clip(new_image, 0, 255).astype(image.dtype)

def gaussian_blur(image: NDArray, kernel_size: int = 11, stdev: float = 0.) -> NDArray:
    new_image = cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=stdev, sigmaY=stdev)
    return new_image

def magnitude_of_gradient(image: NDArray) -> NDArray:
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return magnitude

def minimax_normalization(image: NDArray, dtype: type = np.uint8) -> NDArray:
    image_normalized = (image - image.min()) / (image.max() - image.min()) * 255
    image_normalized = image_normalized.astype(dtype)

    return image_normalized

def resize(image: NDArray,
           width: int | None = None,
           height: int | None = None,
           scale_width: float | None = None,
           scale_height: float | None = None,
           interpolation: int = cv2.INTER_LINEAR) -> NDArray:
    '''
    cv2.INTER_AREA  |  Shrinking  |  Minimizes distortion while downscaling.
    cv2.INTER_LINEAR  |  General resizing  |  Balances speed and quality
    cv2.INTER_CUBIC  |  Enlarging  |  Higher quality for upscaling
    cv2.INTER_NEAREST  |  Fast resizing  |  Quick but lower quality
    '''
    if width is not None or height is not None:
        final_width = width if width is not None else image.shape[1]
        final_height = height if height is not None else image.shape[0]

        new_image = cv2.resize(image, (final_width, final_height), interpolation=interpolation)

    elif scale_width is not None or scale_height is not None:
        final_scale_width = scale_width if scale_width is not None else 1.0
        final_scale_height = scale_height if scale_height is not None else 1.0

        new_image = cv2.resize(image, None, fx = final_scale_width, fy = final_scale_height, interpolation=interpolation)
    else:
        new_image = image.copy()
    return new_image
