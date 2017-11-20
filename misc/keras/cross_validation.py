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


class CrossValidator(object):

    def __init__(self, kfold, loss, image_data_generator=None):
        self.kfold = kfold
        self.image_data_generator = image_data_generator
        self.histories = []

    def _train(self,
               model,
               xs_train, ys_train,
               xs_validation, ys_validation,
               batch_size,
               steps_per_epoch,
               epochs):
        # train
        iter_train = self.image_data_generator.flow(
            xs_train, ys_train,
            batch_size=batch_size,
            shuffle=False,
            seed=None)
        # validation
        iter_validation = self.image_data_generator.flow(
            xs_validation, ys_validation,
            batch_size=batch_size,
            shuffle=False,
            seed=None)
        history = model.fit_generator(
            iter_train,
            steps_per_epoch,
            epochs=epochs,
            validation_data=iter_validation,
            validation_steps=None,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            nitial_epoch=0)
        self.histories.append(history)

    def validate(self,
                 model,
                 path_to_train,
                 path_to_validation,
                 classes,
                 n_splits,
                 steps_per_epoch,
                 epochs,
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

        for train, test in self.kfold.split(xs, ys):
            self._train(
                model,
                xs[train], ys[train],
                xs[test], ys[test],
                batch_size,
                steps_per_epoch,
                epochs)
