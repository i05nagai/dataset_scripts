from . import classify as target
from . import model_helper
from . import settings
from . import util_image
import os


def classify_directory_test():
    path_to_dir = os.path.abspath(os.path.dirname(__file__))
    path_to_test = os.path.join(path_to_dir, './image/validation/open')
    classes = settings.categories
    target_size = settings.target_size
    data_format = 'channels_last'
    color_mode = 'rgb'
    model_name = 'resnet50'

    # model = model_helper.create_model(
    #     model_name, len(classes), target_size)

    # results = target.classify_directory(
    #     model,
    #     path_to_test,
    #     classes,
    #     target_size=target_size,
    #     data_format=data_format,
    #     color_mode=color_mode,
    #     preprocess_function=util_image.preprocess_function)
    # print(results)
