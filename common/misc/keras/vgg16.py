from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import keras
import keras.applications.vgg16 as vgg16
import keras.layers as layers
import math
import numpy as np
import os
import settings
import util


def directory_iterator(
        path_to_image,
        target_size,
        classes,
        batch_size):
    # image generator setttings
    generator = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # data augmentation for image
    dir_iter = generator.flow_from_directory(
        path_to_image,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        shuffle=False)
    return dir_iter


def save_bottleneck_features(
        path_to_image_train,
        path_to_bottleneck_feature_train,
        path_to_image_validation,
        path_to_bottleneck_feature_validation,
        classes,
        target_size,
        batch_size):
    """
    Input images
    and save bottleneck features

    """

    # load vgg16 parameters
    # exclude fully connected layers
    model = vgg16.VGG16(include_top=False, weights='imagenet')

    dir_iter = directory_iterator(
        path_to_image_train, target_size, classes, batch_size)
    print('dir_iter.classes: {0}'.format(dir_iter.classes))
    # save bottoleneck feature for validation data
    num_samples_train = math.ceil(len(dir_iter.classes) / batch_size)
    bottleneck_features_train = model.predict_generator(
        dir_iter, num_samples_train)
    print('bottleneck_features_train.shape: {0}'.format(
        bottleneck_features_train.shape))
    np.save(path_to_bottleneck_feature_train, bottleneck_features_train)

    # data augmentation for image
    dir_iter = directory_iterator(
        path_to_image_validation, target_size, classes, batch_size)
    # save bottoleneck feature for validation data
    num_samples_validation = math.ceil(len(dir_iter.classes) / batch_size)
    bottleneck_features_validation = model.predict_generator(
        dir_iter, num_samples_validation)
    print('bottleneck_features_validation.shape: {0}'.format(
        bottleneck_features_validation.shape))
    np.save(
        path_to_bottleneck_feature_validation, bottleneck_features_validation)


def train_top_model(
        epochs,
        path_to_train,
        path_to_image_train,
        path_to_validation,
        path_to_image_validation,
        batch_size,
        classes,
        target_size,
        path_to_weight=None,
        path_to_history=None):
    """
    Train fully connected layers by bottlenec features

    """
    num_class = len(classes)
    # load train data
    data_train = np.load(path_to_train)
    print('data_train.shape: {0}'.format(data_train.shape))

    model = fully_connected_layers(num_class, data_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # gen labels
    dir_iter = directory_iterator(
        path_to_image_train, target_size, classes, batch_size)
    labels_train = to_categorical(dir_iter.classes)
    dir_iter = directory_iterator(
        path_to_image_validation, target_size, classes, batch_size)
    labels_validation = to_categorical(dir_iter.classes)

    # load validation data
    data_validation = np.load(path_to_validation)
    print('data_validation.shape: {0}'.format(data_validation.shape))
    history = model.fit(data_train,
                        labels_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(data_validation, labels_validation))

    if path_to_weight is not None:
        model.save_weights(path_to_weight)
    if path_to_history is not None:
        util.save_history(history, path_to_history)


def fully_connected_layers(num_class, input_shape):
    # Fully Connected layers
    top_model = keras.models.Sequential()
    top_model.add(layers.Flatten(input_shape=input_shape))
    top_model.add(layers.Dense(256, activation='relu'))
    top_model.add(layers.Dropout(0.5))
    top_model.add(layers.Dense(num_class, activation='sigmoid'))
    return top_model


def combined_model(
        target_size,
        num_class,
        path_to_weight_fc_layer=None,
        path_to_weight_fine_tune=None):
    if path_to_weight_fine_tune is not None:
        path_to_weight_fc_layer = None
    # input_tensor is required
    # https://keras.io/applications/#inceptionv3
    input_tensor = layers.Input(shape=(target_size[0], target_size[1], 3))
    vgg16_model = vgg16.VGG16(
        include_top=False, weights='imagenet', input_tensor=input_tensor)
    # Fully connected layers
    top_model = fully_connected_layers(
        num_class,
        vgg16_model.output_shape[1:])
    # load weights for fc layer
    if path_to_weight_fc_layer is not None:
        top_model.load_weights(path_to_weight_fc_layer)
    # combine vgg16 and our models
    # https://github.com/fchollet/keras/issues/4040
    model = keras.engine.training.Model(
        inputs=vgg16_model.input, outputs=top_model(vgg16_model.output))

    # load weights for fine-tuned combined_model
    if path_to_weight_fine_tune is not None:
        model.load_weights(path_to_weight_fine_tune)

    print('vgg16_model: {0}'.format(vgg16_model))
    print('top_model: {0}'.format(top_model))
    print('model: {0}'.format(model))
    return model


def fine_tune(
        path_to_weight_fc_layer,
        target_size,
        classes,
        epochs,
        batch_size,
        path_to_image_train,
        path_to_image_validation,
        steps_per_epoch_train=None,
        steps_per_epoch_validation=None,
        path_to_weight_fine_tune=None,
        path_to_history_fine_tune=None):
    num_class = len(classes)
    model = combined_model(
        target_size,
        num_class,
        path_to_weight_fc_layer,
        None)

    # show layers
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    datagen_train = image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    dir_iter_train = datagen_train.flow_from_directory(
        path_to_image_train,
        target_size=target_size,
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    datagen_validation = image.ImageDataGenerator(rescale=1.0 / 255.0)
    dir_iter_validation = datagen_validation.flow_from_directory(
        path_to_image_validation,
        target_size=target_size,
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    if steps_per_epoch_train is None:
        steps_per_epoch_train = len(dir_iter_train.classes) / batch_size
    if steps_per_epoch_validation is None:
        steps_per_epoch_validation = (len(dir_iter_validation.classes)
                                      / batch_size)

    # Fine-tuning
    history = model.fit_generator(
        dir_iter_train,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=dir_iter_validation,
        validation_steps=steps_per_epoch_validation)

    if path_to_weight_fine_tune is not None:
        model.save_weights(path_to_weight_fine_tune)
    if path_to_history_fine_tune is not None:
        util.save_history(history, path_to_history_fine_tune)


def predict(path_to_image, target_size, num_class, path_to_weight_fine_tune):
    xs = util.load_single_image(path_to_image, target_size)
    print(xs.shape)
    model = combined_model(
        target_size,
        num_class,
        path_to_weight_fc_layer=None,
        path_to_weight_fine_tune=path_to_weight_fine_tune)
    return model.predict(xs)


def main():
    # paths
    path_to_image_train = settings.path_to_image_train
    path_to_bottleneck_feature_train = settings.path_to_bottleneck_feature_train
    path_to_image_validation = settings.path_to_image_validation
    path_to_bottleneck_feature_validation = settings.path_to_bottleneck_feature_validation
    print(path_to_image_train)
    print(path_to_bottleneck_feature_train)
    print(path_to_image_validation)
    print(path_to_bottleneck_feature_validation)
    batch_size = settings.batch_size
    target_size = settings.target_size
    categories = settings.categories
    print('categories: {0}'.format(categories))
    save_bottleneck_features(
        path_to_image_train,
        path_to_bottleneck_feature_train,
        path_to_image_validation,
        path_to_bottleneck_feature_validation,
        categories,
        target_size,
        batch_size)

    path_to_weight_fc_layer = settings.path_to_weight_fc_layer
    path_to_history_fc_layer = settings.path_to_history_fc_layer
    print('path_to_weight_fc_layer: {0}'.format(path_to_weight_fc_layer))
    print('path_to_history_fc_layer: {0}'.format(path_to_history_fc_layer))

    epochs = settings.epochs
    train_top_model(
        epochs,
        path_to_bottleneck_feature_train,
        path_to_image_train,
        path_to_bottleneck_feature_validation,
        path_to_image_validation,
        batch_size,
        categories,
        target_size,
        path_to_weight=path_to_weight_fc_layer,
        path_to_history=path_to_history_fc_layer)

    path_to_weight_fine_tune = settings.path_to_weight_fine_tune
    path_to_history_fine_tune = settings.path_to_history_fine_tune
    print('path_to_weight_fine_tune: {0}'.format(path_to_weight_fine_tune))
    print('path_to_history_fine_tune: {0}'.format(path_to_history_fine_tune))
    fine_tune(
        path_to_weight_fc_layer,
        target_size,
        categories,
        epochs,
        batch_size,
        path_to_image_train,
        path_to_image_validation,
        path_to_weight_fine_tune=path_to_weight_fine_tune,
        path_to_history_fine_tune=path_to_history_fine_tune)

    # predict by combined model
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    path_to_image = os.path.join(
        path_to_this_dir,
        'image/n07591961_paella/001be5fb5535fe7636b32faff7b81f06aec6ebc8.jpg')
    print('path_to_image: {0}'.format(path_to_image))
    # results = predict(
    #     path_to_image, target_size, num_class, path_to_weight_fine_tune)
    # print(results)


if __name__ == '__main__':
    main()
