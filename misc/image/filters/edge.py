from .. import util

import skimage
import skimage.filters
import numpy as np
import scipy.ndimage as ndimage


DEFAULT_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 6, -1],
     [0, -1, 0]])


def sharpen_kernel(image, kernel=DEFAULT_KERNEL, divisor=2.2):
    image = util.copy(image, dtype='float32')
    for c in range(0, 3):
        image[:, :, c] = ndimage.convolve(image[:, :, c], kernel) / divisor
    image = util.to_valid_image(image)
    return image.astype('uint8', copy=False)


def sharpen(image, radius=0, percent=100, threshold=4.0):
    """sharpen

    :param image:
    :param radius: Standard deviation in pixels.
    :param percent:
    :param threshold:
    """
    unsharp_strength = percent / 100.0

    # Convert to float so that negatives don't cause problems
    image = skimage.img_as_float(image)
    blurred = skimage.filters.gaussian(
        image, radius, multichannel=True, truncate=threshold)
    # image + highpass = image + (image - lowpass)
    sharp = (2 * image - unsharp_strength * blurred) * 255.0
    return util.to_valid_image(sharp)


def smoothe(image, radius=0, threshold=4.0, mask=None):
    """smooth

    :param image:
    :param radius: Standard deviation in pixels.
    :param threshold:
    """
    # Convert to float so that negatives don't cause problems
    image = skimage.img_as_float(image)
    if mask is not None:
        blurred = image[mask]
    else:
        blurred = image
    blurred = skimage.filters.gaussian(
        blurred, radius, multichannel=True, truncate=threshold)
    image[mask] = blurred
    return util.to_valid_image(image * 255.0)


def _mask_circle(img, radius=None, center=None, inverse=False):
    w = img.shape[1]
    h = img.shape[0]
    n = min(w, h)

    if center is None:
        center = (int(h / 2), int(w / 2))
    if radius is None:
        radius = n / 2

    y, x = np.ogrid[-center[0]:h - center[0], -center[1]:w - center[1]]
    mask = x * x + y * y <= radius * radius

    if inverse:
        mask = np.logical_not(mask)
    return mask


def _get_radius(img, shrink=1.0):
    w = img.shape[1]
    h = img.shape[0]
    radius = min(w, h) * shrink
    return radius
