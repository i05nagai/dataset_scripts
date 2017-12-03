import skimage

from . import curve
from . import blend
from .. import histogram
from .. import util


def nashville(img):
    points = [
        (0, 37),
    ]
    curve.intensity_curve_spline(img, 1, points)
    points = [
        (0, 131),
    ]
    curve.intensity_curve_spline(img, 2, points)
    curve.rescale_intensity(img, (0, 236), (0, 255))
    skimage.exposure.adjust_gamma(img, 1.36)
    histogram.adjust_contrast_and_intensity(img, 12, 6)

    # adjust curves colors
    points = [
        (13, 0),
    ]
    curve.intensity_curve_spline(img, 1, points)
    points = [
        (88, 0),
    ]
    curve.intensity_curve_spline(img, 2, points)
    histogram.adjust_contrast_and_intensity(img, 5, -6)

    # adjust curves colors
    points = [
        (0, 4),
    ]
    curve.intensity_curve_spline(img, 0, points)
    points = [
        (0, 14),
    ]
    curve.intensity_curve_spline(img, 2, points)

    fill_color = (255, 218, 173)
    blend.blending(fill_color, img, 'multiply')
    return img


def hefe(img):
    layer = util.copy(img)
    layer = util.rgb_to_rgba(layer, 255)
    # adjust brightness contrast
    img = histogram.adjust_contrast_and_intensity(img, 25, 15)

    # adjust hue saturation
    img = histogram.adjust_hue_saturation_lightness(img, 0, 5, -20, 0)

    print(img)
    img = blend.blending(layer, img, 'overlay')
    print(img)
    img = util.rgba_to_rgb(img)
    # adjust brightness contrast
    img = histogram.adjust_contrast_and_intensity(img, 5, -15)

    fill_color = (181, 181, 181)
    img = blend.blending(fill_color, img, 'multiply')
    img = util.to_valid_image(img)
    return img
