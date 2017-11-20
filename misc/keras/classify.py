from . import util_image
from ..util import filesystem
from . import util


def _classify(
        model, path, classes, target_size, data_format, color_mode,
        preprocess_function=None):
    xs = util_image.load_imgs([path], target_size, data_format, color_mode)

    if preprocess_function is not None:
        for i, x in enumerate(xs):
            xs[i] = preprocess_function(x)

    y = model.predict(xs)[0]
    return util.prediction_to_label(y, classes)


def classify_directory(
        model,
        path_to_dir,
        classes,
        target_size,
        data_format,
        color_mode='rgb',
        preprocess_function=None):
    filepaths = filesystem.get_filepath(path_to_dir, recursive=False)

    results = []
    for path in filepaths:
        result = _classify(
            model, path, classes,
            target_size, data_format, color_mode, preprocess_function)
        results.append(result)
    return result
