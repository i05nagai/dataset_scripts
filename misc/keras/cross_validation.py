import numpy as np
import keras.backend as K
import sklearn.model_selection as model_selection

from . import util_image
from . import util_file


def validate(model_creator,
             xs,
             ys,
             kfold,
             batch_size,
             steps_per_epoch,
             epochs,
             image_data_generator=None):
    histories = []

    for train, test in kfold.split(xs, ys):
        model = model_creator()

        if image_data_generator is not None:
            # train
            iter_train = image_data_generator.flow(
                xs[train], ys[train],
                batch_size=batch_size,
                shuffle=False,
                seed=None)
            # validation
            iter_validation = image_data_generator.flow(
                xs[test], ys[test],
                batch_size=batch_size,
                shuffle=False,
                seed=None)
            history = model.fit_generator(
                iter_train,
                steps_per_epoch,
                epochs=epochs,
                validation_data=iter_validation,
                validation_steps=None,
                class_weight=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                shuffle=True,
                nitial_epoch=0)
        else:
            validation_data = zip(xs[test], ys[test])
            history = model.fit(
                xs[train],
                ys[train],
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=validation_data,
                validation_steps=steps_per_epoch)
        histories.append(history)
        return histories


def kfold_from_directory(
        model_creator,
        path_to_base,
        classes,
        target_size=(256, 256),
        data_format=None,
        batch_size=32,
        steps_per_epoch=1,
        n_splits=2,
        image_data_generator=None):
    if data_format is None:
        data_format = K.image_data_format()

    xs, ys = util_file.get_paths_and_labels(path_to_base, classes)
    xs = util_image.load_imgs(xs, target_size, data_format)
    xs = np.array(xs)
    ys = np.array(ys)

    kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    histories = validate(
        model_creator,
        xs,
        ys,
        kfold,
        batch_size,
        steps_per_epoch,
        n_splits,
        target_size)
    return histories
