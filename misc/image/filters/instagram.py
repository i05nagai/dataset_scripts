import skimage

from . import curve
from . import blend
from .. import histogram


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
    skimage.exposure.adust_gamma(img, 1.36)
    histogram.adjust_contrast_and_lightness(img, 12, 6)

    # adjust curves colors
    points = [
        (13, 0),
    ]
    curve.intensity_curve_spline(img, 1, points)
    points = [
        (88, 0),
    ]
    curve.intensity_curve_spline(img, 2, points)
    histogram.adjust_contrast_and_lightness(img, 5, -6)

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
