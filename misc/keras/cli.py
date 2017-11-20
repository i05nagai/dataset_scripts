import argparse
import os
import keras.applications.resnet50 as resnet50
import keras.engine.training
import keras.layers as layers
import csv

from . import fine_tune
from . import settings
from . import util
from . import util_image


def predict_fine_tune(paths, model_name, classes=None):
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


def predict(paths, model_name, classes=None):
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))

    model = resnet50.ResNet50()
    target_size = settings.target_size

    results = []
    for path_to_image in paths:
        path_to_image = os.path.join(path_to_this_dir, path_to_image)
        print('path_to_image: {0}'.format(path_to_image))
        x = util_image.load_single_image(path_to_image, target_size)
        x = resnet50.preprocess_input(x)
        y = model.predict(x)
        result = resnet50.decode_predictions(y, top=5)

        results.append(result)

    return results


def train(model_name):
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


def main():
    parser = argparse.ArgumentParser(description="keras fine tuner")
    parser.add_argument(
        '--model',
        type=str,
        choices=['vgg16', 'inception_v3', 'resnet50'],
        default='indeption_v3',
        help='help message of this argument')
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help='Train by fine tuning')
    parser.add_argument(
        '--predict',
        metavar="PATH_TO_IMAGES",
        type=str,
        nargs='+',
        help='path to images which you want to predict')
    args = parser.parse_args()

    model_name = args.model

    if args.train:
        train(model_name)
    else:
        images = args.predict
        results = predict(images, model_name)
        try:
            classes = settings.categories
            # header
            outputs = [
                ['path'] + [c for c in classes]
            ]
            # body
            for image, result in zip(images, results):
                outputs.append([image] + [result[c] for c in classes])
            # write to file
            with open("predict_results.csv", "w") as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(outputs)
        except IOError as e:
            print(e)


if __name__ == '__main__':
    main()
