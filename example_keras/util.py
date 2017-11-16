import datetime
import errno
import keras.preprocessing.image as image
import matplotlib.pyplot as plt
import numpy as np
import os


def make_directory(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def predict(model, x, preprocess_input, decode_predictions, top=5):
    input_img = preprocess_input(x)
    predictions = model.predict(input_img)
    return decode_predictions(predictions, top=top)


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        line = '{0}\t{1}\t{2}\t{3}\t{4}\n'
        for i in range(nb_epoch):
            fp.write(line.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))


def load_single_image(path_to_img, target_size):
    img = image.load_img(path_to_img, target_size=target_size)
    # img to numpy array
    x = image.img_to_array(img)
    # (samples, rows, cols, channels)
    xs = np.expand_dims(x, axis=0)
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


def current_datetime_str():
    now = datetime.datetime.now()
    date_string = now.strftime('%Y_%m_%d_%h_%m_%s')
    return date_string


def add_prefix(path, prefix, separator='_'):
    """add_prefix
    Add prefix to filename

    :param path:
    :param prefix:
    :param separator:
    """
    filename = os.path.basename(path)
    filename_new = '{0}{1}{2}'.format(prefix, separator, filename)
    dirpath = os.path.dirname(path)
    path_new = os.path.join(dirpath, filename_new)
    return path_new


def add_suffix(path, suffix, separator='_'):
    """add_suffix
    Add suffix to filename

    :param path:
    :param suffix:
    :param separator:
    """
    filename, ext = os.path.splitext(os.path.basename(path))
    filename_new = '{0}{1}{2}.{3}'.format(filename, separator, suffix, ext)
    dirpath = os.path.dirname(path)
    path_new = os.path.join(dirpath, filename_new)
    return path_new


def prediction_to_label(result, classes):
    """prediction_to_label

    :param result: array of array
    :param classes: array of string
    """

    return [dict(zip(classes, r)) for r in result]
