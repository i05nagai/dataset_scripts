import os
import re


def get_files(path_to_dir):
    # return only file name
    return os.listdir(path_to_dir)


def get_latest_weight(path_to_dir, model_name, basename):
    """get_latest_weight
    2017_11_01_08_28_17_vgg16_weight_fc_layer.h5
    """

    def is_weight_file(filename):
        prefix = r'\d{4}_\d\d_\d\d_\d\d_\d\d_\d\d'
        suffix = '{0}_weigth_{1}.h5'.format(model_name, basename)
        name = '{0}_{1}'.format(prefix, suffix)
        return re.match(name, filename)

    files = get_files(path_to_dir)
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


def get_filepath(path_to_dir, recursive=True):
    if not recursive:
        # return only file name
        return os.listdir(path_to_dir)
    else:
        file_list = []
        for root, subdirs, files in os.walk(path_to_dir):
            for file in files:
                path_to_file = os.path.join(root, file)
                file_list.append(path_to_file)
        return file_list


def get_parent_dir(path_to_dir):
    return os.path.dirname(path_to_dir)
