import os
import re

from ..util import filesystem
from . import util


class PathManager(object):
    """FineTunerPath

    path_to_dir
    - train
    -- img.jpg
    - validation
    -- img.jpg

    """

    HISTORY = 'history'
    WEIGHT = 'weight'

    def __init__(self, path_to_dir, model_name, base_name):
        self.path_to_dir = path_to_dir
        # train/validation
        paths = get_path_train_and_validation(path_to_dir)
        self.train = paths[0]
        self.validation = paths[1]
        # weight
        path_to_weight = os.path.join(path_to_dir, self.WEIGHT)
        self.weight = os.path.join(
            path_to_weight, '{0}.h5'.format(base_name))
        # history
        path_to_history = os.path.join(path_to_dir, self.HISTORY)
        self.history = os.path.join(
            path_to_history, '{0}.txt'.format(base_name))
        # make directory
        filesystem.make_directory(path_to_weight)
        filesystem.make_directory(path_to_history)
        # rename
        self._path_rename(model_name)
        # basename
        self.base_name = base_name

    def _path_rename(self, model_name):
        date_str = util.current_datetime_str()
        prefix = '{0}_{1}'.format(date_str, model_name)

        rename_list = [
            'history',
            'weight',
        ]
        for var_name in rename_list:
            var = getattr(self, var_name)
            new_name = filesystem.add_prefix_to_filename(var, prefix)
            setattr(self, var_name, new_name)

    def get_latest_weight(self, model_name):
        """get_latest_weight
        2017_11_01_08_28_17_vgg16_fine_tuned.h5
        """
        path_to_weight = os.path.join(self.path_to_dir, self.WEIGHT)
        files = filesystem.get_filename(path_to_weight)

        date_str = r'\d{4}_(\d\d_){5}'
        filename = '{0}.h5'.format(self.base_name)
        pattern = '{0}{1}_{2}'.format(date_str, model_name, filename)
        for fname in sorted(files, reverse=True):
            if re.match(pattern, fname):
                return os.path.join(path_to_weight, fname)
        # not found weight file
        raise ValueError('Weight file is not found')

    def get_latest_history():
        """get_latest_history
        2017_11_01_08_28_17_vgg16_history_fc_layer.txt
        """
        pass


def add_model_and_time_as_prefix(path, model_name):
    date_str = util.current_datetime_str()
    prefix = '{0}_{1}'.format(date_str, model_name)

    path = filesystem.add_prefix_to_filename(path, prefix)
    return path


def get_latest_weight(path_to_dir, model_name, basename):
    """get_latest_weight
    2017_11_01_08_28_17_vgg16_weight_fc_layer.h5
    """

    def is_weight_file(filename):
        prefix = r'\d{4}_\d\d_\d\d_\d\d_\d\d_\d\d'
        suffix = '{0}_weigth_{1}.h5'.format(model_name, basename)
        name = '{0}_{1}'.format(prefix, suffix)
        return re.match(name, filename)

    files = filesystem.get_filename(path_to_dir, recursive=False)
    return list(filter(is_weight_file, files))[-1]


def get_latest_history():
    """get_latest_history
    2017_11_01_08_28_17_vgg16_history_fc_layer.txt
    """
    pass


def get_latest_feature():
    """get_latest_feature
    2017_11_01_08_28_17_vgg16_train_feature.npy
    """
    pass


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


def get_paths_and_labels(path_to_base, classes):
    """get_paths_and_labels
    Get paths to images and class labels assumed when we use ImageDataGenerator

    :param path_to_base:
    :param classes:
    """
    path_to_train = os.path.join(path_to_base, 'train')
    path_to_validation = os.path.join(path_to_base, 'validation')

    paths = []
    # train file path
    paths = filesystem.get_filepath(path_to_train)
    # validation file path
    paths += filesystem.get_filepath(path_to_validation)

    labels = get_labels(paths, classes)
    return paths, labels


def get_path_train_and_validation(path_to_base):
    TRAIN = 'train'
    VALIDATION = 'validation'
    train = os.path.join(path_to_base, TRAIN)
    validation = os.path.join(path_to_base, VALIDATION)
    return train, validation
