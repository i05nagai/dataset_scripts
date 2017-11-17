
def load_single_image(path_to_img, target_size):
    img = image.load_img(path_to_img, target_size=target_size)
    # img to numpy array
    x = image.img_to_array(img)
    # (samples, rows, cols, channels)
    xs = np.expand_dims(x, axis=0)
    return xs


def add_image(xs, path_to_image):
    x = load_single_image(path_to_image)
    # (samples, rows, cols, channels)
    xs = np.append(xs, x, axis=0)
    return xs


def prediction_to_label(result, classes):
    """prediction_to_label

    :param result: array of array
    :param classes: array of string
    """

    return [dict(zip(classes, r)) for r in result]
