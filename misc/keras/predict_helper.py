from . import util_image
from ..util import filesystem


def predict_from_directory(
        model,
        path_to_dir,
        target_size=(256, 256),
        data_format=None,
        color_mode='rgb',
        preprocess_function=None,
        decode_predictions=None):
    paths = filesystem.get_filepath(path_to_dir, recursive=False)
    x = util_image.load_imgs(paths, target_size, data_format, color_mode)

    if preprocess_function is not None:
        x = preprocess_function(x)
    y = model.predict(x)
    if decode_predictions is not None:
        y = decode_predictions(y)

    return y
