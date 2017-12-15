from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.preprocessing.image as image
import os
import util
import settings

from . import model_helper


def directory_iterator(
        path_to_image,
        target_size,
        classes,
        batch_size):
    # image generator setttings
    generator = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # data augmentation for image
    dir_iter = generator.flow_from_directory(
        path_to_image,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        shuffle=False)
    return dir_iter


def train(
        model_creator,
        epochs,
        path_to_train,
        path_to_validation,
        batch_size,
        classes,
        target_size,
        path_to_weight,
        path_to_history):
    dir_iter_train = directory_iterator(
        path_to_train, target_size, classes, batch_size)
    dir_iter_validation = directory_iterator(
        path_to_validation, target_size, classes, batch_size)
    model = model_creator()

    # load validation data
    history = model.fit_generator(
        dir_iter_train,
        steps_per_epoch=None,
        epochs=epochs,
        validation_data=dir_iter_validation,
        validation_steps=None)

    model.save_weights(path_to_weight)
    util.save_history(history, path_to_history)


def predict(model, path_to_image, target_size):
    xs = util.load_single_image(path_to_image, target_size)
    return model.predict(xs)


def get_model_creator(base_model, classes, target_size, fine_tune):
    def creator(path_to_weight=None):
        if path_to_weight is None:
            return model_helper.create_model(
                base_model, classes, target_size, fine_tune=fine_tune)
        else:
            model = model_helper.create_model(
                base_model, classes, target_size, fine_tune=fine_tune)
            model.load_weight(path_to_weight)
            return model
    return creator


def main():
    # paths
    path_to_train = settings.path_to_image_train
    path_to_validation = settings.path_to_image_validation
    batch_size = settings.batch_size
    target_size = settings.target_size
    categories = settings.categories
    path_to_weight = settings.path_to_weight_fc_layer
    path_to_history = settings.path_to_history_fc_layer
    epochs = settings.epochs

    train(
        get_model_creator,
        epochs,
        path_to_train,
        path_to_validation,
        batch_size,
        categories,
        target_size,
        path_to_weight=path_to_weight,
        path_to_history=path_to_history)

    # predict by combined model
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    path_to_image = os.path.join(
        path_to_this_dir,
        'path/to/image.jpg')
    print('path_to_image: {0}'.format(path_to_image))
    results = predict(
        path_to_image, target_size, path_to_weight)
    print(results)


if __name__ == '__main__':
    main()
