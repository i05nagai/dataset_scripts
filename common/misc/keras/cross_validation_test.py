from . import cross_validation as target
from . import fine_tune
from . import settings
import sklearn.model_selection as model_selection


def validate_test():
    n_splits = 2
    kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    image_data_generator = ''

    ft_path = fine_tune.FineTunerPath(settings.path_to_base)
    classes = settings.categories
    model_name = 'vgg16'
    model = fine_tune.FineTuner(model_name)
    target_size = settings.target_size

    #h = cv.validate(
    #    model,
    #    ft_path.train,
    #    ft_path.validation,
    #    classes,
    #    n_splits,
    #    target_size)
    #print(h)
