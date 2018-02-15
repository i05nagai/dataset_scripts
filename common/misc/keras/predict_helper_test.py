from . import predict_helper as target


def predict_from_directory_test(faker, mocker):
    model = mocker.Mock()
    path_to_dir = faker.pystr()
    target_size = (faker.pyint(), faker.pytint())
    data_format = faker.pystr()
    color_mode = faker.pystr()
    preprocess_function = None
    decode_predictions = None

    actual = predict_from_directory(
        model, path_to_dir, target_size,
        data_format, color_mode, preprocess_function,
        decode_predictions)
