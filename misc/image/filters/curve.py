import scipy.interpolate
import skimage.exposure
import numpy as np


def validate_interpolate_points(points):
    unique = {}
    for p in points:
        unique[p[0]] = p[1]
    return [(k, v) for k, v in unique.items()]


def curve_spline(points):
    points = [(0, 0)] + points + [(255, 255)]
    points = validate_interpolate_points(points)

    print(points)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    xnew = [i for i in range(0, 256)]
    # linear
    if len(points) <= 3:
        ynew = np.interp(xnew, x, y)
    # spline
    else:
        tck = scipy.interpolate.splrep(x, y, s=0)
        ynew = scipy.interpolate.splev(xnew, tck, der=0)
    return ynew


def intensity_curve_spline(img, channel, points):
    ynew = curve_spline(points)
    for w in range(img.shape[0]):
        for c in range(img.shape[1]):
            img[w, c, channel] = int(ynew[int(img[w, c, channel])])
    return img


def rescale_intensity(img, in_range, out_range):
    for channel in [0, 1, 2]:
        img[:, :, channel] = skimage.exposure.rescale_intensity(
            img[:, :, channel], in_range, out_range)
    return img
