import skimage.io
import skimage.transform
import os
import numpy as np

from ...util import filesystem


def copy(image, dtype=None):
    if dtype is None:
        dtype = image.dtype
    return image.astype(dtype)


def rgb_to_rgba(image, alpha=1.0):
    if image.shape[2] == 4:
        return image
    shape = image.shape
    rgba = np.empty((shape[0], shape[1], 4))
    rgba.fill(alpha)
    rgba[:, :, 0:3] = image
    return rgba


def rgba_to_rgb(image):
    if image.shape[2] == 3:
        return image
    return image[:, :, 0:3]


def empty_image(shape, dtype=None):
    return np.empty(shape, dtype)


def to_valid_pixel(value):
    return max(0, min(value, 255))


def to_valid_image(image):
    return np.clip(image, 0, 255).astype('uint8')


def to_ndarray(image):
    if isinstance(image, str):
        return skimage.io.imread(image)
    return image


def transpose_and_save(path_to_dir, path_to_save_dir, transposer, args={}):
    filenames = filesystem.get_filename(path_to_dir)

    for filename in filenames:
        path = os.path.join(path_to_dir, filename)
        image = to_ndarray(path)
        image = transposer(image, **args)
        path_output = os.path.join(path_to_save_dir, filename)
        skimage.io.imsave(path_output, image)


def resize_image(image, output_shape=(224, 224, 3)):
    image = to_ndarray(image)
    image = skimage.transform.resize(image, output_shape)
    return image
