import numpy as np
import os
import keras.backend as K

from . import util_image
from ..util import filesystem


def get_labels(paths, classes):
    """get_labels

    :param paths:
    :param classes:
    """
    labels = []
    for path in paths:
        dir_name = os.path.basename(os.path.dirname(path))
        if dir_name in classes:
            index = classes.index(dir_name)
            labels.append(index)
        else:
            raise ValueError('directory name must be same as one of classes.'
                             '{0}'.format(dir_name))
    return labels


def get_paths_and_labels(path_to_train, path_to_validation, classes):
    paths = []
    # train file path
    paths = filesystem.get_filepath(path_to_train)
    # validation file path
    paths += filesystem.get_filepath(path_to_validation)

    labels = get_labels(paths, classes)
    return paths, labels


class Report(object):

    def __init__(self, num_cross_validation, loss_functions):
        self.num = [0] * num_cross_validation
        self.num_true = [0] * num_cross_validation
        self.num_loss = [0.0] * num_cross_validation
        self.loss_functions = loss_functions

    def add(self, num_try, predict, output):
        pass


class CrossValidator(object):

    def __init__(self, kfold, loss, image_data_generator=None):
        self.kfold = kfold
        self.image_data_generator = image_data_generator

    def _train(self, xs, target_size, data_format):
        # train
        x = util_image.load_imgs(xs[train], target_size, data_format)
        # train
        array_iter = self.image_data_generator.flow(
            x, ys[train],
            batch_size=batch_size,
            shuffle=False,
            seed=None)

    def _evaluate(self):
        pass

    def validate(self,
                 model,
                 path_to_train,
                 path_to_validation,
                 classes,
                 n_splits,
                 target_size=(256, 256),
                 data_format=None,
                 batch_size=32,
                 steps=None):
        if data_format is None:
            data_format = K.image_data_format()

        xs, ys = get_paths_and_labels(
            path_to_train, path_to_validation, classes)
        xs = util_image.load_imgs(xs, target_size, data_format)
        xs = np.array(xs)
        ys = np.array(ys)
        # report = Report(n_splits)

        num_try = 0
        for train, test in self.kfold.split(xs, ys):
            print(train)
            print(xs[train].shape)
            result = model.evaluate_generator(
                array_iter, steps)
            result = np.argmax(result)
