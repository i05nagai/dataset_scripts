import keras.applications.vgg16 as vgg16
import keras.applications.resnet50 as resnet50
import keras.layers as layers
import keras.engine.training as training


def fine_tuned_model(base_model, num_classes, target_size):

    if isinstance(base_model, str):
        input_tensor = layers.Input(shape=(target_size[0], target_size[1], 3))
        if base_model == 'vgg16':
            base_model = vgg16.VGG16(
                include_top=False, input_tensor=input_tensor)
        if base_model == 'resnet50':
            base_model = resnet50.ResNet50(
                include_top=False,
                input_tensor=input_tensor,
                classes=num_classes)
            x = base_model.outputs[0]
            x = layers.Flatten()(x)
            x = layers.Dense(
                num_classes, activation='softmax', name='fc_layer')(x)
            num_fixed_layers = 173
        else:
            base_model = vgg16.VGG16(include_top=False)

    model = training.Model(base_model.inputs, x, name='fine_tuned_model')
    for layer in model.layers[:num_fixed_layers]:
        layer.trainable = False
    return model
