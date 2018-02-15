import numpy as np
import skimage.io
import os
import PIL.Image
import PIL.ImageFilter

from ..util import filesystem


def to_ndarray(image_pil):
    return np.asarray(image_pil)


def from_ndarray(image_ndarray):
    return PIL.Image.fromarray(np.uint8(image_ndarray))


def read_image(path_to_image):
    img = PIL.Image.open(path_to_image)
    return img


def show_image(image_pil):
    skimage.io.imshow(np.asarray(image_pil))


def save(image_pil, path):
    image_pil.save(path)


def transpose_and_save(path_to_dir, path_to_save_dir, transposer, args={}):
    filenames = filesystem.get_filename(path_to_dir)

    for filename in filenames:
        path_to_input = os.path.join(path_to_dir, filename)
        image = PIL.Image.open(path_to_input)
        image = transposer(image, **args)
        path_output = os.path.join(path_to_save_dir, filename)
        image.save(path_output)


def enhance(image, factor=0.0):
    """enhance

    :param image:
    :param factor: 0.0 to 1.0
    """
    converter = PIL.ImageEnhance.Color(image)
    image_new = converter.enhance(factor)
    return image_new


def sharpen_pil(image, radius=10, percent=200, threshold=5):
    return image.filter(
        PIL.ImageFilter.UnsharpMask(
            radius=radius, percent=percent, threshold=threshold))
