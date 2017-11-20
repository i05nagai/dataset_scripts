import argparse
import os
import keras.applications.resnet50 as resnet50
import keras.engine.training
import keras.layers as layers

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

    target_size = settings.target_size
    input_tensor = layers.Input(shape=(target_size[0], target_size[1], 3))
    model = resnet50.ResNet50(include_top=False, input_tensor=input_tensor)
    print(model.output_shape)

    # path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    # path_to_weight = os.path.join(path_to_this_dir, './image/resnet50_weight.h5')
    top_model = fine_tune._resnet50_top_fully_connected_layers(
        1000, model.output_shape[1:])
    model_combined = keras.engine.training.Model(
        inputs=model.input, outputs=top_model(model.output))

    path_to_weight = '/Users/makotonagai/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    model_combined.load_weights(path_to_weight)

    results = []
    for path_to_image in paths:
        path_to_image = os.path.join(path_to_this_dir, path_to_image)
        print('path_to_image: {0}'.format(path_to_image))
        x = util_image.load_single_image(path_to_image, target_size)
        x = resnet50.preprocess_input(x)
        y = model.predict(x)
        # result = resnet50.decode_predictions(y, top=5)

        results.append(result)
    import pprint
    pprint.pprint(results)
    return results


# def predict(paths, model_name, classes=None):
#     path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
# 
#     model = resnet50.ResNet50()
#     target_size = settings.target_size
# 
#     results = []
#     for path_to_image in paths:
#         path_to_image = os.path.join(path_to_this_dir, path_to_image)
#         print('path_to_image: {0}'.format(path_to_image))
#         x = util_image.load_single_image(path_to_image, target_size)
#         x = resnet50.preprocess_input(x)
#         y = model.predict(x)
#         result = resnet50.decode_predictions(y, top=5)
# 
#         results.append(result)
#     import pprint
#     pprint.pprint(results)
#     return results


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
        predict(images, model_name)


if __name__ == '__main__':
    main()
