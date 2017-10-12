from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import category
import keras
import keras.applications.vgg16 as vgg16
import keras.layers as layers
import label_data
import math
import numpy as np
import os
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
        label_train,
        path_to_image_train,
        path_to_bottleneck_feature_train,
        label_validation,
        path_to_image_validation,
        path_to_bottleneck_feature_validation,
        classes,
        target_size,
        batch_size):
    """
    Input images
    and save bottleneck features

    path_to_train = os.path.join(result_dir, 'train_features.npy')
    path_to_validation = os.path.join(result_dir, 'validation_features.npy')
    """

    # load vgg16 parameters
    # exclude fully connected layers
    model = vgg16.VGG16(include_top=False, weights='imagenet')

    dir_iter = directory_iterator(
        path_to_image_train, target_size, classes, batch_size)
    print('len(dir_iter.filenames): {0}'.format(len(dir_iter.filenames)))
    print('len(dir_iter.classes): {0}'.format(dir_iter.classes))
    # save bottoleneck feature for validation data
    num_samples_train = math.ceil(len(label_train) / batch_size)
    bottleneck_features_train = model.predict_generator(
        dir_iter, num_samples_train)
    print('bottleneck_features_train.shape: {0}'.format(
        bottleneck_features_train.shape))
    np.save(path_to_bottleneck_feature_train, bottleneck_features_train)

    # data augmentation for image
    dir_iter = directory_iterator(
        path_to_image_validation, target_size, classes, batch_size)
    print('len(generator.filenames): {0}'.format(len(dir_iter.filenames)))
    # save bottoleneck feature for validation data
    num_samples_validation = math.ceil(len(label_validation) / batch_size)
    bottleneck_features_validation = model.predict_generator(
        dir_iter, num_samples_validation)
    print('bottleneck_features_validation.shape: {0}'.format(
        bottleneck_features_validation.shape))
    np.save(
        path_to_bottleneck_feature_validation, bottleneck_features_validation)


def train_top_model(
        epochs,
        path_to_train,
        labels_train,
        path_to_validation,
        labels_validation,
        batch_size,
        num_class,
        path_to_weight=None,
        path_to_history=None):
    """
    Train fully connected layers by bottlenec features

    Example
    ========

    path_to_train = os.path.join(result_dir, 'train_features.npy')
    path_to_validation = os.path.join(result_dir, 'validation_features.npy')
    path_to_weight = os.path.join(result_dir, 'bottleneck_fc_model.h5')
    path_to_history = os.path.join(result_dir, 'history_extractor.txt')
    """
    # load train data
    data_train = np.load(path_to_train)
    print('data_train.shape: {0}'.format(data_train.shape))

    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=data_train.shape[1:]))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_class, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

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
        input=vgg16_model.input, output=top_model(vgg16_model.output))

    # load weights for fine-tuned combined_model
    if path_to_weight_fine_tune is not None:
        model.load_weights(path_to_weight_fine_tune)

    print('vgg16_model: {0}'.format(vgg16_model))
    print('top_model: {0}'.format(top_model))
    print('model: {0}'.format(model))
    return model


def fine_tune(
        path_to_weight_fc_layer,
        num_class,
        target_size,
        classes,
        epochs,
        batch_size,
        path_to_image_train,
        num_samples_train,
        path_to_image_validation,
        num_samples_validation,
        path_to_weight_fine_tune,
        path_to_history_fine_tune):
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
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    datagen_validation = image.ImageDataGenerator(rescale=1.0 / 255)

    generator_train = datagen_train.flow_from_directory(
        path_to_image_train,
        target_size=target_size,
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)
    generator_validation = datagen_validation.flow_from_directory(
        path_to_image_validation,
        target_size=target_size,
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # Fine-tuning
    history = model.fit_generator(
        generator_train,
        samples_per_epoch=num_samples_train,
        epochs=epochs,
        validation_data=generator_validation,
        nb_val_samples=num_samples_validation)

    model.save_weights(path_to_weight_fine_tune)
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


def gen_labels(batch_size, num_per_class, num_class):
    labels = []
    for i in range(num_class):
        labels += [i] * num_per_class
    labels = to_categorical(labels)
    print('len(labels): {0}'.format(len(labels)))
    return np.array(labels)


def main():
    labels_train = label_data.labels_train
    labels_validation = label_data.labels_validation
    # paths
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    path_to_image_train = os.path.join(
        path_to_this_dir, 'image/train')
    path_to_bottleneck_feature_train = os.path.join(
        path_to_this_dir, 'image/train_feature.npy')
    path_to_image_validation = os.path.join(
        path_to_this_dir, 'image/validation')
    path_to_bottleneck_feature_validation = os.path.join(
        path_to_this_dir, 'image/validation_feature.npy')
    batch_size = 16
    print(path_to_image_train)
    print(path_to_bottleneck_feature_train)
    print(path_to_image_validation)
    print(path_to_bottleneck_feature_validation)
    print('len(labels_train): {0}'.format(len(labels_train)))
    print('len(labels_validation): {0}'.format(len(labels_validation)))
    target_size = (150, 150)
    num_class = 12
    print('target_size: {0}'.format(target_size))
    print('num_class: {0}'.format(num_class))
    categories = category.read_labels()
    print('categories: {0}'.format(categories))
    save_bottleneck_features(
        labels_train,
        path_to_image_train,
        path_to_bottleneck_feature_train,
        labels_validation,
        path_to_image_validation,
        path_to_bottleneck_feature_validation,
        categories,
        target_size,
        batch_size)

    # train
    data_train = np.load(path_to_bottleneck_feature_train)
    num_samples_train = data_train.shape[0]
    num_per_class_train = int(num_samples_train / num_class)
    labels_train = gen_labels(
        batch_size, num_per_class_train, num_class)
    print('data_train.shape: {0}'.format(data_train.shape))
    print('num_samples_train: {0}'.format(num_samples_train))
    print('num_per_class_train: {0}'.format(num_per_class_train))
    print('labels_train.shape: {0}'.format(labels_train.shape))
    # validation
    data_validation = np.load(path_to_bottleneck_feature_validation)
    num_samples_validation = data_validation.shape[0]
    num_per_class_validation = num_samples_validation // num_class
    labels_validation = gen_labels(
        batch_size, num_per_class_validation, num_class)
    print('data_validation.shape: {0}'.format(data_validation.shape))
    print('num_samples_validation: {0}'.format(num_samples_validation))
    print('num_per_class_validation: {0}'.format(num_per_class_validation))
    print('labels_validation.shape: {0}'.format(labels_validation.shape))

    path_to_weight_fc_layer = os.path.join(
        path_to_this_dir, 'image/weight_fc_layer.h5')
    path_to_history_fc_layer = os.path.join(
        path_to_this_dir, 'image/history_fc_layer.txt')
    print('path_to_weight_fc_layer: {0}'.format(path_to_weight_fc_layer))
    print('path_to_history_fc_layer: {0}'.format(path_to_history_fc_layer))

    epochs = 50
    # train_top_model(
    #     epochs,
    #     path_to_bottleneck_feature_train,
    #     labels_train,
    #     path_to_bottleneck_feature_validation,
    #     labels_validation,
    #     batch_size,
    #     num_class,
    #     path_to_weight=path_to_weight_fc_layer,
    #     path_to_history=path_to_history_fc_layer)

    path_to_weight_fine_tune = os.path.join(
        path_to_this_dir, 'image/train_weight_fine_tune.h5')
    path_to_history_fine_tune = os.path.join(
        path_to_this_dir, 'image/train_history_fine_tune.txt')
    print('path_to_weight_fine_tune: {0}'.format(path_to_weight_fine_tune))
    print('path_to_history_fine_tune: {0}'.format(path_to_history_fine_tune))
    # fine_tune(
    #     path_to_weight_fc_layer,
    #     num_class,
    #     target_size,
    #     categories,
    #     epochs,
    #     batch_size,
    #     path_to_image_train,
    #     num_samples_train,
    #     path_to_image_validation,
    #     num_samples_validation,
    #     path_to_weight_fine_tune,
    #     path_to_history_fine_tune)

    # predict by combined model
    path_to_image = os.path.join(
        path_to_this_dir,
        'image/n07591961_paella/001be5fb5535fe7636b32faff7b81f06aec6ebc8.jpg')
    print('path_to_image: {0}'.format(path_to_image))
    # results = predict(
    #     path_to_image, target_size, num_class, path_to_weight_fine_tune)
    # print(results)


if __name__ == '__main__':
    main()
