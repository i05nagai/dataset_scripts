import scipy.interpolate
import skimage.exposure


def curve_spline(points):
    points = [(0, 0)] + points + [(255, 255)]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    tck = scipy.interpolate.splrep(x, y, s=0)
    xnew = [i for i in range(0, 256)]
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    return ynew


def intensity_curve_spline(img, channel, points):
    ynew = curve_spline(points)
    shape = img.shape
    for w in range(shape[0]):
        for c in range(shape[1]):
            img[w, c, channel] = int(ynew[int(img[w, c, channel])])
    return img


def reslace_intensity(img, in_range, out_range):
    for channel in [0, 1, 2]:
        img[:, :, channel] = skimage.exposure.rescale_intensity(
            img[:, :, channel], in_range, out_range)
    return img
