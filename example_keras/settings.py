import json
import os


PATH_TO_CONFIG = 'category.json'


def read_settings(path=PATH_TO_CONFIG):
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    abspath = os.path.join(path_to_this_dir, path)
    try:
        with open(abspath, 'r') as f:
            config = json.load(f)
    except IOError as e:
        print(e)
    return config


def get_labels(config, sort=True):
    labels = []
    for category in config['categories']:
        labels += [category['label']]
    if sort:
        labels = sorted(labels)
    return labels


def read_labels(path=PATH_TO_CONFIG, sort=True):
    config = read_settings(path)
    return get_labels(config, sort)


target_size = (224, 224)
batch_size = 16
categories = read_labels()
epochs = 50

# path
path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
path_to_base = os.path.join(path_to_this_dir, 'image')
path_to_image_train = os.path.join(path_to_base, 'train')
path_to_image_validation = os.path.join(path_to_base, 'validation')
path_to_bottleneck_feature_train = os.path.join(
    path_to_base, 'train_feature.npy')
path_to_bottleneck_feature_validation = os.path.join(
    path_to_base, 'validation_feature.npy')
path_to_weight_fc_layer = os.path.join(path_to_base, 'weight_fc_layer.h5')
path_to_history_fc_layer = os.path.join(path_to_base, 'history_fc_layer.txt')
path_to_weight_fine_tune = os.path.join(
    path_to_base, 'train_weight_fine_tune.h5')
path_to_history_fine_tune = os.path.join(
    path_to_base, 'train_history_fine_tune.txt')
