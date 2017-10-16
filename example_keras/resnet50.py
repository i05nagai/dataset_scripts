import keras.applications.resnet50 as resnet50
from keras.preprocessing import image
import numpy as np
import sys


def load_single_image(path_to_img):
    img = image.load_img(path_to_img, target_size=(224, 224))
    # 読み込んだPIL形式の画像をarrayに変換
    x = image.img_to_array(img)
    # 4次元テンソル (samples, rows, cols, channels) に変換
    x = np.expand_dims(x, axis=0)
    return x


def main():
    model = resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None)

    filename = sys.argv[1]
    x = load_single_image(filename)
    input_img = resnet50.preprocess_input(x)
    predictions = model.predict(input_img)
    results = resnet50.decode_predictions(predictions, top=5)
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
