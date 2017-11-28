import skimage.io
import skimage.transform
import os

from ..util import filesystem


def to_ndarray(image):
    if isinstance(image, str):
        return skimage.io.imread(image)
    return image


def transpose_and_save(path_to_dir, path_to_save_dir, transposer, args={}):
    filenames = filesystem.get_filepath(path_to_dir)

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
