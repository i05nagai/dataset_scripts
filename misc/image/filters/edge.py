from PIL import ImageFilter


def shapen(image, radius=10, percent=200, threshold=5):
    return image.filter(
        ImageFilter.UnsharpMask(
            radius=radius, percent=percent, threshold=threshold))
