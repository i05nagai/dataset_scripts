import numpy as np
import skimage.io
import os
import PIL.Image

from ..util import filesystem


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
