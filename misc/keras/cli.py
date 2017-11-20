import argparse
import os
import keras.applications.resnet50 as resnet50
import csv

from . import fine_tune
from . import settings
from . import util_image


def write_predict_to_csv(xs, ys, classes):
    # header
    outputs = [
        ['path'] + [c for c in classes]
    ]
    # body
    for x, y in zip(xs, ys):
        outputs.append([x] + [y[c] for c in classes])
    try:
        # write to file
        with open("predict_results.csv", "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(outputs)
    except IOError as e:
        print(e)


def predict_fine_tune(paths, model_name, classes=None):
    classes = settings.categories
    target_size = settings.target_size
    path_to_base = settings.path_to_base

    results = fine_tune.predict(
        paths,
        model_name,
        classes,
        target_size,
        path_to_base)

    import pprint
    pprint.pprint(results)


def train_fine_tune(model_name):
    path_to_base = settings.path_to_base
    classes = settings.categories
    batch_size = settings.batch_size
    target_size = settings.target_size
    epochs = settings.epochs

    fine_tune.train(
        model_name,
        classes,
        batch_size,
        target_size,
        epochs,
        path_to_base)


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


def main():
    # fine tune modelのtrainingとprediction
    # fine tuneじゃないmodelのtrainとprediction
    # fine tuneのmodelのcross validationをしたい
    # 両方のmodelのclassify_directoryをしたい
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
        train_fine_tune(model_name)
    else:
        paths = args.predict
        classes = settings.categories
        results = predict_fine_tune(paths, model_name, classes)
        write_predict_to_csv(paths, results, classes)
    classes = settings.categories


if __name__ == '__main__':
    main()
