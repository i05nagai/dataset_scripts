import keras.applications.imagenet_utils as imagenet_utils
import numpy as np
import keras.backend as K
import keras.preprocessing.image as image

from . import util


def get_image_shape(target_size, data_format, color_mode='rgb'):
    if color_mode == 'rgb':
        if data_format == 'channels_last':
            image_shape = target_size + (3,)
        else:
            image_shape = (3,) + target_size
    else:
        if data_format == 'channels_last':
            image_shape = target_size + (1,)
        else:
            image_shape = (1,) + target_size
    return image_shape


def load_single_image(path_to_img, target_size):
    img = image.load_img(path_to_img, target_size=target_size)
    # img to numpy array
    x = image.img_to_array(img)
    # (samples, rows, cols, channels)
    xs = np.expand_dims(x, axis=0)
    return xs


def load_imgs(
        paths, target_size, data_format='channels_last', color_mode='rgb'):
    """load_imgs_as_batch

    :param paths:
    :param target_size:
    :param data_format:
    :param color_mode:

    :return: 4D numpy array
    :rtype:
    """
    data_format = util.get_data_format(data_format)
    image_shape = get_image_shape(target_size, data_format, color_mode)
    xs = np.zeros((len(paths),) + image_shape, dtype=K.floatx())
    for i, path in enumerate(paths):
        img = image.load_img(path, target_size=target_size)
        x = image.img_to_array(img)
        xs[i] = x
    return xs


def add_image(xs, path_to_image):
    x = load_single_image(path_to_image)
    # (samples, rows, cols, channels)
    xs = np.append(xs, x, axis=0)
    return xs


def draw_image_from_array(x):
    img = image.array_to_img(x)
    plt.imshow(img)
    plt.show()


def preprocess_function(x):
    """preprocess_function
    for imagenet input from ImageDataGenerator

    :param x: rank 3 tensor. (w, h, c)

    :return:
    :rtype: rank 3 tensor
    """
    xs = np.expand_dims(x, axis=0)
    xs = imagenet_utils.preprocess_input(xs)
    xs = np.squeeze(xs, axis=0)
    return xs
