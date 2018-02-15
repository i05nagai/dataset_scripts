import keras.optimizers


def train_from_directory(faker, mocker):
    model = mocker.Mock()
    num_classes = 2
    target_size = (faker.pyint(), faker.pyint())
    classes = settings.categories
    batch_size = faker.pyint()
    epochs = faker.pyint()
    steps_per_epoch = faker.pyint()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy'])

    actual = train_from_directory()
