import skimage
import numpy as np
import curve


def _filter_nashville(img):
    filt = np.empty(img.shape)


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
    #adjust brightness contrast
    #adjust curves colors
    points = [
        (0, 4),
    ]
    curve.intensity_curve_spline(img, 0, points)
    points = [
        (0, 14),
    ]
    curve.intensity_curve_spline(img, 2, points)

    w = img.shape[0]
    h = img.shape[1]

    fill_color = (255,218,173)

