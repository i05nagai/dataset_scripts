from . import fine_tune
from . import settings
from ..util import filesystem
from . import util

import os


def train_test():
    model_name = 'resnet50'
    # paths
    path_to_base = settings.path_to_base

    ft_path = fine_tune.FineTunerPath(path_to_base)
    ft_path._path_rename(model_name)

    classes = settings.categories
    batch_size = settings.batch_size
    target_size = settings.target_size
    epochs = settings.epochs

    fine_tuner = fine_tune.FineTuner(model_name)
    fine_tuner.train(
        ft_path,
        classes,
        target_size,
        batch_size,
        epochs)


def predict_test():
    model_name = 'resnet50'
    paths = filesystem.get_filepath('image/validation/private')
    classes = settings.categories

    fine_tuner = fine_tune.FineTuner(model_name)
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))

    num_class = len(settings.categories)
    target_size = settings.target_size

    ft_path = fine_tune.FineTunerPath(settings.path_to_base)
    path_to_weight_fine_tune = ft_path.get_latest_weight(model_name)
    print('path_to_weight_fine_tune: {0}'.format(path_to_weight_fine_tune))

    results = []
    for image in paths:
        path_to_image = os.path.join(path_to_this_dir, image)
        print('path_to_image: {0}'.format(path_to_image))
        result = fine_tuner.predict(
            path_to_image, target_size, num_class, path_to_weight_fine_tune)

        if classes is not None:
            result = util.prediction_to_label(result, classes)

        results.append(result)
    print(results)
    return results
