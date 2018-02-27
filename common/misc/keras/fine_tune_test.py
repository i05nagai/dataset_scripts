from . import fine_tune as target
from . import settings
from ...util import filesystem

import os


def train_test():
    model_name = 'resnet50'
    path_to_base = settings.path_to_base
    classes = settings.categories
    batch_size = settings.batch_size
    target_size = settings.target_size
    epochs = settings.epochs

    # target.train(
    #     model_name,
    #     classes,
    #     batch_size,
    #     target_size,
    #     epochs,
    #     path_to_base)


def predict_test():
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    path_to_predict = os.path.join(
        path_to_this_dir, 'image/validation/private')
    paths = filesystem.get_filepath(path_to_predict)
    model_name = 'resnet50'

    classes = settings.categories
    target_size = settings.target_size
    path_to_base = settings.path_to_base

    # results = target.predict(
    #     paths,
    #     model_name,
    #     classes,
    #     target_size,
    #     path_to_base)

    # import pprint
    # pprint.pprint(results)
