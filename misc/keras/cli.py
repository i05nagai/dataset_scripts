import os
import csv

from . import settings
import click


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()
@click.argument('paths', nargs=-1, type=tuple)
@click.option(
    '--model_name',
    type=click.Choice(['resnet50', 'vgg16', 'inception_v3']),
    default='resnet50')
@click.option('--data_dir', type=str)
@click.option('--fine_tune', is_flag=True, default=False)
def predict(paths, model_name, data_dir, fine_tune):
    from ..util import filesystem

    paths = list(paths)
    if data_dir is not None:
        paths += filesystem.get_filepath(data_dir)

    if fine_tune:
        classes = settings.categories
        results = predict_fine_tune(paths, model_name, classes)
        print('paths: {0}'.format(paths))
        write_predict_to_csv(paths, results, classes)


@cli.command()
@click.option(
    '--model_name',
    type=click.Choice(['resnet50', 'vgg16', 'inception_v3']),
    default='resnet50')
@click.option('--data_dir')
@click.option('--fine_tune', default=False)
def train(model_name, data_dir, fine_tune):
    if fine_tune:
        train_fine_tune(model_name)
    else:
        pass


def write_predict_to_csv(xs, ys, classes):
    # header
    outputs = [
        ['path'] + [c for c in classes]
    ]
    print(xs)
    print(ys)
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


def train_fine_tune(model_name):
    from . import fine_tune
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
