from . import util
from . import util_image
from ...util import filesystem


def _classify(
        model,
        paths,
        classes,
        target_size=(256, 256),
        data_format=None,
        color_mode='rgb',
        preprocess_function=None):
    xs = util_image.load_imgs(paths, target_size, data_format, color_mode)

    if preprocess_function is not None:
        for i, x in enumerate(xs):
            xs[i] = preprocess_function(x)

    y = model.predict(xs)
    return util.prediction_to_label(y, classes)


def classify_directory(
        model,
        path_to_dir,
        classes,
        target_size=(256, 256),
        data_format=None,
        color_mode='rgb',
        preprocess_function=None):
    data_format = util.get_data_format(data_format)
    filepaths = filesystem.get_filepath(path_to_dir, recursive=False)

    result = _classify(
        model,
        filepaths,
        classes,
        target_size=target_size,
        data_format=data_format,
        color_mode=color_mode,
        preprocess_function=preprocess_function)
    return result
