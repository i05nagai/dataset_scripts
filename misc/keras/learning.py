from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Tuple
import keras.preprocessing.image as image
import keras.optimizers
import os

from . import model_helper
from . import settings
from . import util


def directory_iterator(
        path_to_image: str,
        target_size: Tuple[int, int],
        classes,
        batch_size: int):
    # image generator setttings
    generator = image.ImageDataGenerator(
        rotation_range=90,
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
        shuffle=True)
    return dir_iter


def train(
        model_creator,
        epochs: int,
        path_to_train: str,
        path_to_validation: str,
        batch_size: int,
        classes,
        target_size: Tuple[int, int],
        path_to_weight: str,
        path_to_history: str) -> None:

    dir_iter_train = directory_iterator(
        path_to_train, target_size, classes, batch_size)
    dir_iter_validation = directory_iterator(
        path_to_validation, target_size, classes, batch_size)
    model = model_creator()

    steps_per_epoch = len(dir_iter_train.classes) / batch_size
    validation_steps = len(dir_iter_validation.classes) / batch_size

    history = model.fit_generator(
        dir_iter_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=dir_iter_validation,
        validation_steps=validation_steps)

    model.save_weights(path_to_weight)
    util.save_history(history, path_to_history)


def predict(
        model,
        path_to_image,
        target_size,
        path_to_weight):
    """predict

    :param model:
    :param path_to_image:
    :param target_size:
    :param path_to_weight:

    Example
    ========

    >>> model = model_creator(path_to_weight)
    >>> for image in images:
    >>>     path_to_image = os.path.join(
    >>>         path_to_base, image)
    >>>     results = predict(
    >>>         model,
    >>>         path_to_image,
    >>>         target_size,
    >>>         path_to_weight)
    """
    xs = util.load_single_image(path_to_image, target_size, preprocess_input)
    return model.predict(xs)


def predict_on_batch(
        model_creator,
        paths_to_image,
        batch_size,
        target_size,
        path_to_weight):
    model = model_creator(path_to_weight)
    results = []

    for i in range(0, len(paths_to_image), batch_size):
        paths = paths_to_image[i:i + batch_size]
        xs = util.load_image_batch(paths, target_size, preprocess_input)
        result_batch = model.predict_on_batch(xs)
        results.extend(result_batch)

    return results


def preprocess_input(x):
    return x / 255.0


def get_model_creator(
        base_model,
        classes,
        target_size: Tuple[int, int],
        train_all_layers: bool):
    def creator(path_to_weight=None):
        if path_to_weight is None:
            model = model_helper.create_model(
                base_model,
                classes,
                target_size,
                train_all_layers=train_all_layers)
        else:
            model = model_helper.create_model(
                base_model,
                classes,
                target_size,
                train_all_layers=train_all_layers)
            model.load_weights(path_to_weight)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])
        return model
    return creator


def main():
    path_to_base = settings.path_to_base
    # paths
    path_to_train = os.path.join(path_to_base, 'train')
    path_to_validation = os.path.join(path_to_base, 'validation')
    path_to_weight = os.path.join(path_to_base, 'weight.h5')
    path_to_history = os.path.join(path_to_base, 'history.txt')

    batch_size = settings.batch_size
    target_size = settings.target_size
    classes = settings.categories
    epochs = settings.epochs

    base_model = 'vgg16'
    model_creator = get_model_creator(
        base_model, classes, target_size, train_all_layers=True)

    train(
        model_creator,
        epochs,
        path_to_train,
        path_to_validation,
        batch_size,
        classes,
        target_size,
        path_to_weight=path_to_weight,
        path_to_history=path_to_history)

    # predict by combined model
    images = [
    ]
    images = [os.path.join(path_to_base, image) for image in images]

    results = predict_on_batch(
        model_creator,
        images,
        target_size,
        path_to_weight)
    print(results)


if __name__ == '__main__':
    main()
