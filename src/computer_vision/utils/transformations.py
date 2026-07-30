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
    return new_image.astype(image.dtype)

def salt_and_pepper(image : NDArray, salt_prob : float, pepper_prob : float):
    row, col = image.shape[0:2]
    image_s_p = image.copy()
    n_points = row * col

    salt_points = int(salt_prob * n_points)

    x_salt = np.random.randint(0, row, size=salt_points)
    y_salt = np.random.randint(0, col, size=salt_points)


    pepper_points = int(pepper_prob * n_points)

    x_pepper = np.random.randint(0, row, size=pepper_points)
    y_pepper = np.random.randint(0, col, size=pepper_points)


    if len(image_s_p.shape) == 3:
        image_s_p[x_salt, y_salt] = [255,255,255] 
        image_s_p[x_pepper, y_pepper] = [0,0,0] 
    else:
        image_s_p[x_salt, y_salt] = 255
        image_s_p[x_salt, y_salt] = 0

    return image_s_p

def gaussian_noise_conv(image : NDArray,  mean: float, stdev: float, kernel_size : int):
    noise = np.random.normal(loc=mean, scale=stdev, size=image.shape)

    box_kernel = np.ones((kernel_size, kernel_size), dtype = np.float32) / kernel_size**2
    noise_conv = cv2.filter2D(src=noise, ddepth=-1, kernel=box_kernel)

    new_image = image + noise_conv

    return np.clip(new_image, 0, 255).astype(image.dtype)


def salt_and_pepper_conv(image : NDArray, salt_prob : float, pepper_prob : float, kernel_size : int, sigma : float | None = None):

    if sigma == None:
        sigma = (kernel_size // 2) / 3
    # Gaussian kernel has floating point values. Some values will be higher than 255 or lower than 0, so float32 is necessary
    image_s_p = image.copy().astype(np.float32)
    h, w = image_s_p.shape[0], image_s_p.shape[1]
    n_points = h * w

    is_color = len(image.shape) == 3

    # Random Points
    salt_points = int(salt_prob * n_points)

    x_salt = np.random.randint(0, h, size=salt_points)
    y_salt = np.random.randint(0, w, size=salt_points)

    pepper_points = int(pepper_prob * n_points)

    x_pepper = np.random.randint(0, h, size=pepper_points)
    y_pepper = np.random.randint(0, w, size=pepper_points)

    # Setting Gaussian Kernel
    kernel = cv2.getGaussianKernel(ksize = kernel_size, sigma = sigma)
    kernel = kernel @ kernel.T
    
    mid_down = kernel_size // 2
    mid_up = kernel_size - mid_down

    factor = 1 / kernel[mid_down,mid_down]
    kernel = factor * kernel.astype(np.float32) * 20
    
    # Padding
    pad_tuple = ((mid_down, mid_up), (mid_down,mid_up), (0,0)) if is_color else (mid_down, mid_up)
    image_s_p = np.pad(image_s_p, pad_width=pad_tuple)

    x_salt_pad = x_salt + mid_down
    y_salt_pad = y_salt + mid_down
    x_pepper_pad = x_pepper + mid_down
    y_pepper_pad = y_pepper + mid_down

    # Value to substitute
    kernel = np.repeat(kernel[:,:,np.newaxis], 3, axis=2) if is_color else kernel

    # Adding values to padded image in int16

    # Salt Values
    for xs, ys in zip(x_salt_pad, y_salt_pad):
        image_s_p[xs-mid_down:xs+mid_up, ys-mid_down:ys+mid_up] += kernel

    #Pepper Values
    for xp, yp in zip(x_pepper_pad, y_pepper_pad):
        # Pepper Values are substracted
        image_s_p[xp-mid_down:xp+mid_up, yp-mid_down:yp+mid_up] -= kernel

    return np.clip(image_s_p[mid_down:mid_down+h, mid_down:mid_down+w], 0, 255).astype(image.dtype) # clip to 0 to 255 limits and go back to initial dtype


def time_decaying_artifacts(image : NDArray, old_noise : NDArray | None, event_probability : float, additive : bool,  decay_factor : float, kernel_size : int, kernel_factor : float, sigma : float | None = None):

    if sigma == None:
        sigma = (kernel_size // 2) / 3
    # Gaussian kernel has floating point values. Some values will be higher than 255 or lower than 0, so float32 is necessary
    image_s_p = image.copy().astype(np.float32)
    h, w = image_s_p.shape[0], image_s_p.shape[1]

    is_color = len(image.shape) == 3

    if old_noise is None:
        old_noise = np.zeros(image_s_p.shape, dtype = image_s_p.dtype)
    # Diminishing old artifacts
    elif image_s_p.shape == old_noise.shape:
        old_noise = decay_factor*old_noise  
        if additive:
            image_s_p += old_noise
        else:
            image_s_p -= old_noise
    else:
        raise IndexError("Old noise doesn't have the same size of the original image")

    prob = np.random.random(1)[0]

    if prob > event_probability:
        print("Event didn't happen")
        return np.clip(image_s_p, 0, 255).astype(image.dtype), old_noise
    
   
    x_artifact = np.random.randint(0, h, size=1)[0]
    y_artifact = np.random.randint(0, w, size=1)[0]

    # Setting Gaussian Kernel
    kernel = cv2.getGaussianKernel(ksize = kernel_size, sigma = sigma)
    kernel = kernel @ kernel.T

    mid_down = kernel_size // 2
    mid_up = kernel_size - mid_down

    factor = 1 / kernel[mid_down,mid_down]
    kernel = factor * kernel.astype(np.float32) * kernel_factor
    
    # Padding
    pad_tuple = ((mid_down, mid_up), (mid_down,mid_up), (0,0)) if is_color else (mid_down, mid_up)
    image_s_p = np.pad(image_s_p, pad_width=pad_tuple)
    noise = np.pad(old_noise, pad_width=pad_tuple)

    x_artifact_pad = x_artifact + mid_down
    y_artifact_pad = y_artifact + mid_down

    # Value to substitute
    kernel = np.repeat(kernel[:,:,np.newaxis], 3, axis=2) if is_color else kernel

    # New event Values
    if additive:
        image_s_p[x_artifact_pad-mid_down:x_artifact_pad+mid_up, y_artifact_pad-mid_down:y_artifact_pad+mid_up] += kernel
    else:
        image_s_p[x_artifact_pad-mid_down:x_artifact_pad+mid_up, y_artifact_pad-mid_down:y_artifact_pad+mid_up] -= kernel

    noise[x_artifact_pad-mid_down:x_artifact_pad+mid_up, y_artifact_pad-mid_down:y_artifact_pad+mid_up] += kernel

    return np.clip(image_s_p[mid_down:mid_down+h, mid_down:mid_down+w], 0, 255).astype(image.dtype), np.clip(noise[mid_down:mid_down+h, mid_down:mid_down+w], 0, 255).astype(image.dtype) # clip to 0 to 255 limits and go back to initial dtype

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

def crop(image : NDArray,
         proportion : tuple[int,int] | float):
    
    height, width = image.shape[0], image.shape[1]

    if isinstance(proportion, tuple):
        target_ratio = proportion[0]/proportion[1]
    else:
        target_ratio = proportion


    current_ratio = width/height

    print(f"Transforming from proportion {current_ratio} to {target_ratio}")

    if target_ratio>current_ratio:
        new_height = int(width / target_ratio)
        remove = width - new_height
        offset = remove//2

        new_image = image[offset:new_height+offset, :]
    else:

        new_width = int(height * target_ratio)
        remove = width - new_width
        offset = remove//2

        new_image = image[:, offset:new_width+offset]

    return new_image
