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
path_to_base = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'data/images/private')
path_to_base = os.path.abspath(path_to_base)
path_to_dir = os.path.abspath(os.path.dirname(__file__))
