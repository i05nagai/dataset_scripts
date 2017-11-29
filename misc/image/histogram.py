from . import util
import skimage.exposure
import skimage.io
import skimage.color


def adjust_lightness(image, shift=5):
    image = util.to_ndarray(image)
    image = skimage.color.rgb2hsv(image)
    for c in range(image.shape[0]):
        for r in range(image.shape[1]):
            image[c, r, 2] = util.to_valid_pixel(image[c, r, 2] + shift)
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
