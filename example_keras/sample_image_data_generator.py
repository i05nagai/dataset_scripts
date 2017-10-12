import keras.preprocessing.image as image
import util


def image_data_augumentation(
        x,
        prefix,
        save_to_dir='./image/restaurant/',
        batch_size=32):
    # import keras.preprocessing.image as image
    img_datagen = image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1.0 / 255)
    gen = img_datagen.flow(x,
                           batch_size=batch_size,
                           save_to_dir=save_to_dir,
                           save_prefix=prefix,
                           save_format='jpg')
    for i in range(batch_size):
        gen.next()


def generate_img(
        x, img_datagen, batch_size, save_to_dir, prefix):
    gen = img_datagen.flow(x,
                           batch_size=batch_size,
                           save_to_dir=save_to_dir,
                           save_prefix=prefix,
                           save_format='jpg')
    for i in range(batch_size):
        gen.next()


def featurewise_center(
        x, batch_size, save_to_dir, prefix='featurewise_center'):
    img_datagen = image.ImageDataGenerator(
        featurewise_center=True)
    img_datagen.fit(x)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def samplewise_center(
        x, batch_size, save_to_dir, prefix='samplewise_center'):
    img_datagen = image.ImageDataGenerator(
        samplewise_center=True)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def featurewise_std_normalization(
        x, batch_size, save_to_dir, prefix='featurewise_std_normalization'):
    img_datagen = image.ImageDataGenerator(
        featurewise_std_normalization=True)
    img_datagen.fit(x)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def samplewise_std_normalization(
        x, batch_size, save_to_dir, prefix='samplewise_std_normalization'):
    img_datagen = image.ImageDataGenerator(
        samplewise_std_normalization=True)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def zca_whitening(
        x, batch_size, save_to_dir, prefix='zca_whitening'):
    img_datagen = image.ImageDataGenerator(
        zca_whitening=True, zca_epsilon=1e-3)
    img_datagen.fit(x)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def rotation_range(
        x, batch_size, save_to_dir, prefix='rotation_range'):
    img_datagen = image.ImageDataGenerator(
        rotation_range=1.0 / batch_size)
    img_datagen.fit(x)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def width_shift_range(
        x, batch_size, save_to_dir, prefix='width_shift_range'):
    img_datagen = image.ImageDataGenerator(
        width_shift_range=5.0)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def height_shift_range(
        x, batch_size, save_to_dir, prefix='height_shift_range'):
    img_datagen = image.ImageDataGenerator(
        height_shift_range=5.0)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def shear_range(
        x, batch_size, save_to_dir, prefix='shear_range'):
    img_datagen = image.ImageDataGenerator(
        shear_range=0.2)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def zoom_range(
        x, batch_size, save_to_dir, prefix='zoom_range'):
    img_datagen = image.ImageDataGenerator(
        zoom_range=0.2)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def channel_shift_range(
        x, batch_size, save_to_dir, prefix='channel_shift_range'):
    img_datagen = image.ImageDataGenerator(
        channel_shift_range=0.2)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def fill_mode(
        x, batch_size, save_to_dir, prefix='fill_mode'):
    fill_type = [
        'constant',
        'nearest',
        'reflect',
        'wrap',
    ]
    for ftype in fill_type:
        prefix_new = '{0}_{1}'.format(prefix, ftype)
        img_datagen = image.ImageDataGenerator(
            fill_mode=0.2)
        generate_img(x, img_datagen, batch_size, save_to_dir, prefix_new)


def horizontal_flip(
        x, batch_size, save_to_dir, prefix='horizontal_flip'):
    img_datagen = image.ImageDataGenerator(
        horizontal_flip=True)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def vertical_flip(
        x, batch_size, save_to_dir, prefix='vertical_flip'):
    img_datagen = image.ImageDataGenerator(
        vertical_flip=True)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def rescale(
        x, batch_size, save_to_dir, prefix='rescale'):
    img_datagen = image.ImageDataGenerator(
        rescale=1.0 / 255)
    generate_img(x, img_datagen, batch_size, save_to_dir, prefix)


def main():
    path_to_image1 = 'image/sample/0a6c77c4901a6fb84b20b203843d5675366c48c7.jpg'
    path_to_image2 = 'image/sample/0a8d19b049995a1d5850042787cdb397a5145ffc.jpg'
    xs = util.load_single_image(path_to_image1)
    xs = util.add_image(xs, path_to_image2)

    batch_size = 16
    save_to_dir = './image/sample_output/'
    # featurewise_center(xs, batch_size, save_to_dir)
    # samplewise_center(xs, batch_size, save_to_dir)
    # featurewise_std_normalization(xs, batch_size, save_to_dir)
    # samplewise_std_normalization(xs, batch_size, save_to_dir)
    zca_whitening(xs, batch_size, save_to_dir)
    # rotation_range(xs, batch_size, save_to_dir)
    # width_shift_range(xs, batch_size, save_to_dir)
    # height_shift_range(xs, batch_size, save_to_dir)
    # shear_range(xs, batch_size, save_to_dir)
    # zoom_range(xs, batch_size, save_to_dir)
    # channel_shift_range(xs, batch_size, save_to_dir)
    # horizontal_flip(xs, batch_size, save_to_dir)
    # vertical_flip(xs, batch_size, save_to_dir)
    # rescale(xs, batch_size, save_to_dir)


if __name__ == '__main__':
    main()
