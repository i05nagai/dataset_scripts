import datetime
import keras.backend as K
import keras.preprocessing.image as image
import numpy as np


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


def current_datetime_str():
    now = datetime.datetime.now()
    date_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    return date_string


def prediction_to_label(result, classes):
    """prediction_to_label

    :param result: array of array
    :param classes: array of string
    """

    return [dict(zip(classes, r)) for r in result]


def get_data_format(data_format):
    if data_format is None:
        return K.image_data_format()
    return data_format
