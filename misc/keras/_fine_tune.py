import keras.applications.vgg16 as vgg16
import keras.applications.resnet50 as resnet50
import keras.layers as layers
import keras.engine.training as training


def create_model(base_model, classes, target_size):

    if isinstance(base_model, str):
        input_tensor = layers.Input(shape=(target_size[0], target_size[1], 3))
        if base_model == 'vgg16':
            base_model = vgg16.VGG16(
                include_top=False, input_tensor=input_tensor)
        if base_model == 'resnet50':
            base_model = resnet50.ResNet50(
                include_top=False,
                input_tensor=input_tensor)
            x = base_model.outputs[0]
            batch_input_shape = base_model.output_shape
            x = layers.Flatten(batch_input_shape=batch_input_shape)(x)
            print(x)
            x = layers.Dense(
                len(classes), activation='softmax', name='fc_layer')(x)
            print(x)
            num_fixed_layers = 173
        else:
            base_model = vgg16.VGG16(include_top=False)

    model = training.Model(base_model.inputs[0], x, name='fine_tuned_model')
    for layer in model.layers[:num_fixed_layers]:
        layer.trainable = False
    return model
