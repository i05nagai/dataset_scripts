import skimage
import skimage.filters
from .. import util


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


def smoothe(image, radius=0, threshold=4.0):
    """smooth

    :param image:
    :param radius: Standard deviation in pixels.
    :param threshold:
    """
    # Convert to float so that negatives don't cause problems
    image = skimage.img_as_float(image)
    blurred = skimage.filters.gaussian(
        image, radius, multichannel=True, truncate=threshold)
    return util.to_valid_image(blurred * 255.0)
