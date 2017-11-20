from ..util import filesystem
import os
import re


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
