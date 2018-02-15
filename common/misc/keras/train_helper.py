from . import fine_tune
from . import util
from . import util_file


def train_from_directory(
        model,
        path_to_base,
        classes,
        target_size=(256, 256),
        batch_size=32,
        epochs=1,
        steps_per_epoch_train=None,
        steps_per_epoch_validation=None,
        path_to_history=None,
        path_to_weight=None):

    path_to_train, path_to_validation = util_file.get_path_train_and_validation(
        path_to_base)

    # iter
    iter_train = fine_tune.gen_directory_iterator(
        path_to_train, target_size, classes, batch_size, True)
    iter_validation = fine_tune.gen_directory_iterator(
        path_to_validation, target_size, classes, batch_size, True)

    # if iter used, steps is needed
    if steps_per_epoch_train is None:
        steps_per_epoch_train = len(iter_train.classes) / batch_size
    if steps_per_epoch_validation is None:
        steps_per_epoch_validation = len(iter_validation.classes) / batch_size

    history = model.fit_generator(
        iter_train,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_steps=steps_per_epoch_validation,
        validation_data=iter_validation)

    if path_to_history is not None:
        util.save_history(history, path_to_history)
    if path_to_weight is not None:
        model.save_weights(path_to_weight)
