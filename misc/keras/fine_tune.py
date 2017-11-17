from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
# import keras.applications.inception_resnet_v2 as inception_resnet_v2
import keras
import keras.applications.inception_v3 as inception_v3
import keras.applications.resnet50 as resnet50
import keras.applications.vgg16 as vgg16
import keras.layers as layers
import math
import numpy as np
import os
import re

from ..util import filesystem
from . import util


class FineTunerPath(object):
    """FineTunerPath

    path_to_dir
    - train
    -- img.jpg
    - validation
    -- img.jpg

    """

    TRAIN = 'train'
    VALIDATION = 'validation'
    HISTORY = 'history'
    WEIGHT = 'weight'
    FEATURE = 'feature'

    def __init__(self, path_to_dir):
        self.path_to_dir = path_to_dir
        # train/validation
        self.train = os.path.join(path_to_dir, self.TRAIN)
        self.validation = os.path.join(path_to_dir, self.VALIDATION)
        # feature
        path_to_feature = os.path.join(path_to_dir, self.FEATURE)
        self.feature_bottleneck_train = os.path.join(
            path_to_feature, 'bottleneck_train.npy')
        self.feature_bottleneck_validation = os.path.join(
            path_to_feature, 'bottleneck_validation.npy')
        # weight
        path_to_weight = os.path.join(path_to_dir, self.WEIGHT)
        self.weight_fc_layer = os.path.join(
            path_to_weight, 'fc_layer.h5')
        self.weight_fine_tuned = os.path.join(
            path_to_weight, 'fine_tuned.h5')
        # history
        path_to_history = os.path.join(path_to_dir, self.HISTORY)
        self.history_fc_layer = os.path.join(
            path_to_history, 'fc_layer.txt')
        self.history_fine_tuned = os.path.join(
            path_to_history, 'fine_tuned.txt')
        # make directory
        filesystem.make_directory(path_to_feature)
        filesystem.make_directory(path_to_weight)
        filesystem.make_directory(path_to_history)

    def _path_rename(self, model_name):
        date_str = util.current_datetime_str()
        prefix = '{0}_{1}'.format(date_str, model_name)

        rename_list = [
            'feature_bottleneck_train',
            'feature_bottleneck_validation',
            'history_fc_layer',
            'history_fine_tuned',
            'weight_fc_layer',
            'weight_fine_tuned',
        ]
        for var_name in rename_list:
            var = getattr(self, var_name)
            new_name = filesystem.add_prefix_to_filename(var, prefix)
            setattr(self, var_name, new_name)

    def get_latest_weight(self, model_name):
        """get_latest_weight
        2017_11_01_08_28_17_vgg16_weight_fc_layer.h5
        """
        path_to_weight = os.path.join(self.path_to_dir, self.WEIGHT)
        files = filesystem.get_filename(path_to_weight)

        date_str = r'(\d\d_){6}'
        filename = 'fine_tuned.h5'
        pattern = '{0}{1}_{2}.h5'.format(date_str, model_name, filename)
        for fname in sorted(files, reverse=True):
            print(fname)
            if re.match(pattern, fname):
                return os.path.join(path_to_weight, fname)
        # not found weight file
        raise ValueError('Weight file is not found')

    def get_latest_history():
        """get_latest_history
        2017_11_01_08_28_17_vgg16_history_fc_layer.txt
        """
        pass

    def get_latest_feature():
        """get_latest_feature
        2017_11_01_08_28_17_vgg16_train_feature.npy
        """
        pass


def _resnet50_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.Flatten(input_shape=input_shape))
    top_model.add(layers.Dense(num_class, activation='softmax', name='fc'))
    return top_model


def _vgg16_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.Flatten(input_shape=input_shape))
    top_model.add(layers.Dense(256, activation='relu'))
    top_model.add(layers.Dropout(0.5))
    top_model.add(layers.Dense(num_class, activation='sigmoid', name='fc'))
    return top_model


def _inception_resnet_v2_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.GlobalAveragePooling2D(
        input_shape=input_shape, name='avg_pool'))
    top_model.add(layers.Dense(num_class, activation='softmax', name='fc'))
    return top_model


def _inception_v3_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.GlobalAveragePooling2D(
        input_shape=input_shape, name='avg_pool'))
    top_model.add(layers.Dense(num_class, activation='softmax', name='fc'))
    return top_model


def directory_iterator(
        path_to_image,
        target_size,
        classes,
        batch_size,
        shuffle):
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
        class_mode='binary',
        shuffle=shuffle)
    return dir_iter


class FineTuner(object):

    def __init__(self, model_name):
        if model_name == 'resnet50':
            self.model = resnet50.ResNet50
            self.top_model = _resnet50_top_fully_connected_layers
            self.num_fixed_layers = 173
        elif model_name == 'inception_resnet_v2':
            # self.model = inception_resnet_v2.InceptionResNetV2
            self.top_model = _inception_resnet_v2_top_fully_connected_layers
            self.num_fixed_layers = 310
        elif model_name == 'inception_v3':
            self.model = inception_v3.InceptionV3
            self.top_model = _inception_v3_top_fully_connected_layers
            self.num_fixed_layers = 310
        else:
            self.model = vgg16.VGG16
            self.top_model = _vgg16_top_fully_connected_layers
            self.num_fixed_layers = 15
        self.model_name = model_name

    def combined_model(
            self,
            target_size,
            num_class,
            path_to_weight_fc_layer=None,
            path_to_weight_fine_tune=None):
        if path_to_weight_fine_tune is not None:
            path_to_weight_fc_layer = None
        # input_tensor is required
        # https://keras.io/applications/#inceptionv3
        input_tensor = layers.Input(shape=(target_size[0], target_size[1], 3))
        model = self.model(
            include_top=False, weights='imagenet', input_tensor=input_tensor)
        # Fully connected layers
        top_model = self.top_model(
            num_class,
            model.output_shape[1:])
        # load weights for fc layer
        if path_to_weight_fc_layer is not None:
            top_model.load_weights(path_to_weight_fc_layer)
        # combine our models
        # https://github.com/fchollet/keras/issues/4040
        model_trained = keras.engine.training.Model(
            inputs=model.input, outputs=top_model(model.output))

        # load weights for fine-tuned combined_model
        if path_to_weight_fine_tune is not None:
            model_trained.load_weights(path_to_weight_fine_tune)

        print('bottleneck_model: {0}'.format(model))
        print('top_model: {0}'.format(top_model))
        print('model: {0}'.format(model))
        return model_trained

    def save_bottleneck_features(
            self, ft_path, classes, target_size, batch_size):
        """
        Input images
        and save bottleneck features
        """
        # load vgg16 parameters
        # exclude fully connected layers
        model = self.model(include_top=False, weights='imagenet')

        dir_iter = directory_iterator(
            ft_path.train, target_size, classes, batch_size, False)
        print('dir_iter.classes: {0}'.format(dir_iter.classes))
        # save bottoleneck feature for validation data
        num_samples_train = math.ceil(len(dir_iter.classes) / batch_size)
        bottleneck_features_train = model.predict_generator(
            dir_iter, num_samples_train)
        print('bottleneck_features_train.shape: {0}'.format(
            bottleneck_features_train.shape))
        np.save(ft_path.feature_bottleneck_train, bottleneck_features_train)

        # data augmentation for image
        dir_iter = directory_iterator(
            ft_path.validation, target_size, classes, batch_size, False)
        # save bottoleneck feature for validation data
        num_samples_validation = math.ceil(len(dir_iter.classes) / batch_size)
        bottleneck_features_validation = model.predict_generator(
            dir_iter, num_samples_validation)
        print('bottleneck_features_validation.shape: {0}'.format(
            bottleneck_features_validation.shape))
        np.save(
            ft_path.feature_bottleneck_validation,
            bottleneck_features_validation)

    def train_top_model(
            self, ft_path, epochs, batch_size, classes, target_size):
        """
        Train fully connected layers by bottlenec features
        """
        num_class = len(classes)
        # load train data
        data_train = np.load(ft_path.feature_bottleneck_train)
        print('data_train.shape: {0}'.format(data_train.shape))

        model = self.top_model(num_class, data_train.shape[1:])
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        # gen labels
        dir_iter = directory_iterator(
            ft_path.train, target_size, classes, batch_size, False)
        labels_train = to_categorical(dir_iter.classes)
        dir_iter = directory_iterator(
            ft_path.validation, target_size, classes, batch_size, False)
        labels_validation = to_categorical(dir_iter.classes)

        # load validation data
        data_validation = np.load(ft_path.feature_bottleneck_validation)
        print('data_validation.shape: {0}'.format(data_validation.shape))
        history = model.fit(data_train,
                            labels_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(data_validation, labels_validation))

        model.save_weights(ft_path.weight_fc_layer)
        util.save_history(history, ft_path.history_fc_layer)

    def fine_tune(
            self,
            ft_path,
            target_size,
            classes,
            epochs,
            batch_size,
            steps_per_epoch_train=None,
            steps_per_epoch_validation=None):
        num_class = len(classes)
        model = self.combined_model(
            target_size,
            num_class,
            ft_path.weight_fc_layer,
            None)

        for layer in model.layers[:self.num_fixed_layers]:
            layer.trainable = False

        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        dir_iter_train = directory_iterator(
            ft_path.train, target_size, classes, batch_size, True)
        dir_iter_validation = directory_iterator(
            ft_path.validation, target_size, classes, batch_size, True)

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

        model.save_weights(ft_path.weight_fine_tuned)
        util.save_history(history, ft_path.history_fine_tuned)

    def train(self, ft_path, classes, target_size, batch_size, epochs):

        self.save_bottleneck_features(
            ft_path, classes, target_size, batch_size)
        print('bottleneck features are saved')

        self.train_top_model(
            ft_path, epochs, batch_size, classes, target_size)
        print('top model are trained')

        self.fine_tune(
            ft_path, target_size, classes, epochs, batch_size)

    def predict(
            self, path_to_image, target_size, num_class, ft_path):
        xs = util.load_single_image(path_to_image, target_size)
        xs = preprocess_input(xs)
        model = self.combined_model(
            target_size,
            num_class,
            path_to_weight_fc_layer=None,
            path_to_weight_fine_tune=ft_path.weight_fine_tuned)
        return model.predict(xs)
