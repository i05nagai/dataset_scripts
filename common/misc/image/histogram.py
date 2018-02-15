from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from . import util
import skimage.exposure
import skimage.io
import skimage.color
import numpy as np


def adjust_overexposure(image, threshold=1e-3):
    # intensity = get_intensity(image)
    return None


def adjust_color(image, factor):
    image_new = util.empty_image(image.shape, image.dtype)
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    image_new[:, :, 0] = image[:, :, 0] * 299.0 / 1000.0
    image_new[:, :, 1] = image[:, :, 1] * 587.0 / 1000.0
    image_new[:, :, 2] = image[:, :, 2] * 114.0 / 1000.0
    image_new = image_new + factor * (image - image_new)
    image_new = util.to_valid_image(image_new)
    return image_new


def adjust_hue_saturation_lightness(
        image, hue_range=0, hue_offset=0, saturation=0, lightness=0):
    # hue is mapped to [0, 1] from [0, 360]
    if hue_offset not in range(-180, 180):
        raise ValueError('Hue should be within (-180, 180)')
    if saturation not in range(-100, 100):
        raise ValueError('Saturation should be within (-100, 100)')
    if lightness not in range(-100, 100):
        raise ValueError('Lightness should be within (-100, 100)')
    image = skimage.color.rgb2hsv(image)
    offset = ((180 + hue_offset) % 180) / 360.0
    image[:, :, 0] = image[:, :, 0] + offset
    image[:, :, 1] = image[:, :, 1] + saturation / 200.0
    image[:, :, 2] = image[:, :, 2] + lightness / 200.0
    image = skimage.color.hsv2rgb(image) * 255.0
    image = util.to_valid_image(image)
    return image


def is_low_contrast(image, contrast=1, intensity=0):
    image = util.to_ndarray(image)
    image = util.copy(image)
    image = contrast * image + intensity
    image = util.to_valid_image(image)
    return image


def adjust_contrast_and_intensity(image, contrast=0, intensity=0):
    image = util.to_ndarray(image)
    image = util.copy(image, dtype='uint16')
    factor = int(259.0 * (contrast + 255.0) / (255.0 * (259.0 - contrast)))
    image = factor * (image - 128) + 128 + intensity
    image = util.to_valid_image(image)
    return image.astype('uint8', copy=False)


def adjust_contrast_css(image, contrast=0):
    """adjust_contrast_css
    CSS specification compatible contrast adjustment

    https://www.w3.org/TR/filter-effects-1/#contrastEquivalent

    :param image:
    :param contrast:
    """
    image = util.to_ndarray(image)
    image = util.copy(image, dtype='uint16')
    image = contrast * image + (0.5 - (contrast * 0.5)) * 255
    image = util.to_valid_image(image)
    return image.astype('uint8', copy=False)


def adjust_brightness_css(image, brightness=0):
    """adjust_brightness_css
    CSS specification compatible brightness adjustment

    https://www.w3.org/TR/filter-effects-1/#brightnessEquivalent

    :param image:
    :param brightness:
    """
    image = util.to_ndarray(image)
    image = util.copy(image, dtype='float64')
    image = image * brightness
    image = util.to_valid_image(image)
    return image.astype('uint8', copy=False)


def adjust_saturation_css(image, saturation=0):
    """adjust_saturation_css
    CSS specification compatible saturate adjustment

    https://www.w3.org/TR/filter-effects-1/#feColorMatrixElement

    :param image:
    :param brightness:
    """
    s = saturation
    color_matrix = np.asarray([
        [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
    ]).T
    image = util.to_ndarray(image)
    image = util.copy(image, dtype='float64')
    image = np.tensordot(image, color_matrix, 1)
    image = util.to_valid_image(image)
    return image.astype('uint8', copy=False)


def adjust_lightness(image, shift=0.0):
    image = util.to_ndarray(image)
    image = util.copy(image)
    image = skimage.color.rgb2ycbcr(image)
    image[:, :, 0] = image[:, :, 0] + shift
    image = util.to_valid_image(image)
    image = skimage.color.ycbcr2rgb(image)
    return image


def adjust_gamma(image, gamma=1.0):
    image = util.to_ndarray(image)
    image = util.copy(image)
    for c in range(image.shape[2]):
        image[:, :, c] = skimage.exposure.adjust_gamma(
            image[:, :, c], gamma=gamma)
    return image


def is_overexposure(image, threshold=10):
    intensity = get_intensity(image)
    p1, p50, p98 = np.percentile(intensity, (1, 50, 99))
    if p1 > threshold:
        return True
    else:
        return False


def stretch_contrast(image, threshold=10):
    image = util.to_ndarray(image)
    if is_overexposure(image, threshold):
        intensity = get_intensity(image)
        p1, p99 = np.percentile(intensity, (1, 99))
        return skimage.exposure.rescale_intensity(image, in_range=(p1, p99))
    else:
        return image


def equalize_histogram(image):
    image = util.to_ndarray(image)
    image_ycbcr = skimage.color.rgb2ycbcr(image)
    # image_ycbcr[:, :, 0] = skimage.exposure.rescale_intensity(
    #     image_ycbcr[:, :, 0], (16, 235), (0, 255))
    # image_ycbcr[:, :, 0] = skimage.exposure.equalize_hist(image_ycbcr[:, :, 0]) * 255
    image_ycbcr[:, :, 0] = skimage.exposure.equalize_hist(image_ycbcr[:, :, 0])
    image_ycbcr[:, :, 0] = skimage.exposure.rescale_intensity(
        image_ycbcr[:, :, 0], (0, 255), (16, 235))
    image = skimage.color.ycbcr2rgb(image_ycbcr)
    return image


def equalize_histogram_adaptive(image):
    image = util.to_ndarray(image)
    image = util.copy(image)
    image = skimage.exposure.equalize_hist(image)
    return image


def equalize_histogram_ycbcr(image, color_sp='ycbcr'):
    valid_color_sp = ['ycbcr']
    if color_sp not in valid_color_sp:
        raise ValueError('color_sp must be one of {0}'.format(valid_color_sp))

    image = util.to_ndarray(image)

    if color_sp == 'ycbcr':
        image = skimage.color.rgb2ycbcr(image)

    image[:, :, 0] = skimage.exposure.equalize_hist(image[:, :, 0])

    if color_sp == 'ycbcr':
        image = skimage.color.ycbcr2rgb(image)
    return image


def get_intensity(image):
    # RGB
    if image.shape[2] == 3:
        b = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    # RGBA
    else:
        b = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
    intensity = np.tensordot(image, b, 1)
    return util.to_valid_image(intensity)


def get_pdf(image):
    hist, nbins = skimage.exposure.cumulative_distribution(image)
    pdf = np.diff(hist)
    return pdf, nbins
