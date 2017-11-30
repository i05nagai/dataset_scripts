from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from . import util
import skimage.exposure
import skimage.io
import skimage.color


def adjust_contrast_and_intensity(image, contrast=0, intensity=0):
    image = util.to_ndarray(image)
    image = util.copy(image, dtype='uint16')
    factor = int(259.0 * (contrast + 255.0) / (255.0 * (259.0 - contrast)))
    image = factor * (image - 128) + 128 + intensity
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


def adjust_gamma(image, gamma=0.5):
    image = util.to_ndarray(image)
    image = skimage.exposure.adjust_gamma(image, gamma=gamma)
    return image


def equalize_histogram(image):
    image = util.to_ndarray(image)
    image = skimage.exposure.equalize_hist(image)
    return image


def equalize_histogram_adaptive(image):
    image = util.to_ndarray(image)
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
