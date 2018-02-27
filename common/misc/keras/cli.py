import os
import csv

import click

from . import settings


MODELS = [
    'resnet50',
    'vgg16',
    'inception_v3',
]
DEFAULT_MODEL = 'vgg16'


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command(help='Predict')
@click.argument('paths', nargs=-1, type=tuple)
@click.option(
    '--model_name',
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL)
@click.option('--data_dir', type=str)
@click.option('--fine_tune', is_flag=True, default=False)
def predict(paths, model_name, data_dir, fine_tune):
    from ..util import filesystem
    from ..ml import score

    paths = list(paths)
    if data_dir is not None:
        paths += filesystem.get_filepath(data_dir)

    path_csv = 'predict_result.csv'
    path_json = 'predict_result.json'
    if fine_tune:
        classes = settings.categories
        results = predict_fine_tune(paths, model_name, classes)
        # save as csv
        write_predict_to_csv(path_csv, paths, results, classes)
        # save as json
        results = score.get_top_n_prediction_from_file(path_csv, 1)
        filesystem.save_as_json(path_json, results)


@cli.command(help='Train models')
@click.option(
    '--model_name',
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL)
@click.option('--data_dir')
@click.option('--fine_tune', is_flag=True, default=False)
def train(model_name, data_dir, fine_tune):
    if fine_tune:
        train_fine_tune(model_name, data_dir)
    else:
        pass


@cli.command(help='Run cross validation')
@click.option(
    '--model_name',
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL)
@click.option('--data_dir')
@click.option('--fine_tune', is_flag=True, default=False)
def cross_validation(model_name, data_dir, fine_tune):
    if fine_tune:
        cross_validation_fine_tune(model_name)
    else:
        pass


@cli.command(help='Train models')
@click.option(
    '--model_name',
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL)
@click.option('--data_dir')
@click.option('--fine_tune', is_flag=True, default=False)
def train_old(model_name, data_dir, fine_tune):
    if fine_tune:
        train_fine_tune_old(model_name, data_dir)
    else:
        pass


@cli.command(help='Predict')
@click.argument('paths', nargs=-1, type=tuple)
@click.option(
    '--model_name',
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL)
@click.option('--data_dir', type=str)
@click.option('--fine_tune', is_flag=True, default=False)
def predict_old(paths, model_name, data_dir, fine_tune):
    from ..util import filesystem
    from ..ml import score

    paths = list(paths)
    if data_dir is not None:
        paths += filesystem.get_filepath(data_dir)

    path_csv = 'predict_result.csv'
    path_json = 'predict_result.json'
    if fine_tune:
        classes = settings.categories
        results = predict_fine_tune_old(paths, model_name, classes)
        # save as csv
        write_predict_to_csv(path_csv, paths, results, classes)
        # save as json
        results = score.get_top_n_prediction_from_file(path_csv, 1)
        filesystem.save_as_json(path_json, results)


def cross_validation_fine_tune(model_name):
    from . import model_helper
    from . import cross_validation as cv
    import keras.optimizers

    image_data_generator = None
    n_splits = 2
    steps_per_epoch = None

    # from settings
    classes = settings.categories
    path_to_base = settings.path_to_base
    target_size = settings.target_size
    batch_size = settings.batch_size
    epochs = settings.epochs

    base_model = model_name

    def model_creator():
        model = model_helper.create_model(base_model, classes, target_size)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])
        return model

    cv.kfold_from_directory(
        model_creator,
        path_to_base,
        classes,
        target_size=target_size,
        data_format=None,
        batch_size=batch_size,
        epochs=epochs,
        n_splits=n_splits,
        steps_per_epoch=steps_per_epoch,
        image_data_generator=image_data_generator)


def write_predict_to_csv(path, xs, ys, classes):
    # header
    outputs = [
        ['path'] + [c for c in classes]
    ]
    # body
    for x, y in zip(xs, ys):
        outputs.append([x] + [y[c] for c in classes])
    try:
        # write to file
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(outputs)
    except IOError as e:
        print(e)


def predict_fine_tune(paths, model_name, classes=None):
    from . import model_helper
    from . import predict_helper
    from . import util_image
    from . import util_file
    from . import util

    classes = settings.categories
    target_size = settings.target_size
    path_to_base = settings.path_to_base

    path_manager = util_file.PathManager(path_to_base, model_name, 'fine_tune')
    path_to_weight = path_manager.get_latest_weight(model_name)

    model = model_helper.create_model(
        model_name, classes, target_size, fine_tune=True)
    model.load_weights(path_to_weight)

    def decode_predictions(y):
        return util.prediction_to_label(y, classes)

    results = predict_helper.predict_from_path(
        model,
        paths,
        target_size,
        preprocess_function=util_image.preprocess_function,
        decode_predictions=decode_predictions)

    return results


def predict_fine_tune_old(paths, model_name, classes=None):
    from . import fine_tune
    classes = settings.categories
    target_size = settings.target_size
    path_to_base = settings.path_to_base

    results = fine_tune.predict(
        paths,
        model_name,
        classes,
        target_size,
        path_to_base)

    return results


def train_fine_tune_old(model_name, data_dir):
    from . import fine_tune

    path_to_base = settings.path_to_base
    if data_dir is not None:
        path_to_base = data_dir
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


def train_fine_tune(model_name, data_dir):
    from . import train_helper
    from . import model_helper
    from . import util_file
    import keras.optimizers

    path_to_base = settings.path_to_base
    if data_dir is not None:
        path_to_base = data_dir

    classes = settings.categories
    batch_size = settings.batch_size
    target_size = settings.target_size
    epochs = settings.epochs

    model = model_helper.create_model(model_name, classes, target_size)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy'])

    path_manager = util_file.PathManager(path_to_base, model_name, 'fine_tune')
    path_to_history = path_manager.history
    path_to_weight = path_manager.weight

    train_helper.train_from_directory(
        model,
        path_to_base,
        classes,
        target_size=target_size,
        batch_size=batch_size,
        epochs=epochs,
        path_to_history=path_to_history,
        path_to_weight=path_to_weight)


def predict_normal(paths, model_name, classes=None):
    import keras.applications.resnet50 as resnet50
    from . import util_image
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
    # fine tune model training and rediction
    # not fine tune model, trainining and prediction
    # fine tune model cross validationを
    # both model classify_directory
    cli()


if __name__ == '__main__':
    main()
