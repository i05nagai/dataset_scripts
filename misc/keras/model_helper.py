import keras.applications.vgg16 as vgg16
import keras.applications.resnet50 as resnet50
import keras.layers as layers
import keras.engine.training as training
import keras.models


def _create_model_fine_tune(base_model, classes, target_size):

    if isinstance(base_model, str):
        input_tensor = layers.Input(shape=(target_size[0], target_size[1], 3))
        if base_model == 'vgg16':
            base_model = vgg16.VGG16(
                include_top=False, input_tensor=input_tensor)
            top_model = vgg16_top_fully_connected_layers(
                len(classes), base_model.output_shape[1:])
            num_fixed_layers = 15
        elif base_model == 'resnet50':
            base_model = resnet50.ResNet50(
                include_top=False,
                input_tensor=input_tensor)
            top_model = resnet50_top_fully_connected_layers(
                len(classes), base_model.output_shape[1:])
            num_fixed_layers = 173
        else:
            base_model = vgg16.VGG16(include_top=False)
            top_model = vgg16_top_fully_connected_layers(
                len(classes), base_model.output_shape[1:])
            num_fixed_layers = 15

    model = training.Model(
        base_model.input,
        top_model(base_model.output),
        name='fine_tuned_model')
    for layer in model.layers[:num_fixed_layers]:
        layer.trainable = False
    return model


def resnet50_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.Flatten(input_shape=input_shape))
    top_model.add(layers.Dense(num_class, activation='softmax', name='fc'))
    return top_model


def vgg16_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.Flatten(input_shape=input_shape))
    top_model.add(layers.Dense(256, activation='relu'))
    top_model.add(layers.Dropout(0.5))
    top_model.add(layers.Dense(num_class, activation='sigmoid', name='fc'))
    return top_model


def inception_resnet_v2_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.GlobalAveragePooling2D(
        input_shape=input_shape, name='avg_pool'))
    top_model.add(layers.Dense(num_class, activation='softmax', name='fc'))
    return top_model


def inception_v3_top_fully_connected_layers(num_class, input_shape):
    top_model = keras.models.Sequential()
    top_model.add(layers.GlobalAveragePooling2D(
        input_shape=input_shape, name='avg_pool'))
    top_model.add(layers.Dense(num_class, activation='softmax', name='fc'))
    return top_model


def create_model(base_model, classes, target_size, fine_tune=True):
    if fine_tune:
        model = _create_model_fine_tune(base_model, classes, target_size)
    else:
        model = _create_model_fine_tune(base_model, classes, target_size)
    return model
