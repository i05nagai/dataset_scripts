from . import _fine_tune
from . import settings
from . import fine_tune
from . import util
from . import util_image

import keras.optimizers
import keras.applications.resnet50 as resnet50
import os


def train_from_images():
    base_model = 'resnet50'
    num_classes = 2
    target_size = settings.target_size
    classes = settings.categories
    batch_size = settings.batch_size
    epochs = 1
    steps_per_epoch = 1

    model = _fine_tune.fine_tuned_model(base_model, num_classes, target_size)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy'])

    ft_path = fine_tune.FineTunerPath(settings.path_to_base)

    iter_train = fine_tune.gen_directory_iterator(
        ft_path.train, target_size, classes, batch_size, True)
    iter_validation = fine_tune.gen_directory_iterator(
        ft_path.validation, target_size, classes, batch_size, True)

    history = model.fit_generator(
        iter_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=steps_per_epoch,
        validation_data=iter_validation)
    path_to_history = 'hoge.txt'
    util.save_history(history, path_to_history)

    path_to_weight = 'weight.h5'
    model.save_weights(path_to_weight)
    model.load_weights(path_to_weight)

    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))

    results = []
    paths = [
        'image/test/s_003o.jpg',
        'image/test/s_0n5d.jpg',
    ]
    for path_to_image in paths:
        path_to_image = os.path.join(path_to_this_dir, path_to_image)
        print('path_to_image: {0}'.format(path_to_image))
        x = util_image.load_single_image(path_to_image, target_size)
        x = resnet50.preprocess_input(x)
        y = model.predict(x)
        result = y

        results.append(result)
    import pprint
    pprint.pprint(results)
