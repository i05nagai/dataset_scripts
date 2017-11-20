from . import cross_validation as target
import sklearn.model_selection as model_selection
from . import settings
from . import fine_tune


def validate_test():
    n_splits = 2
    kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    loss = {'binary_crossentropy'}
    image_data_generator = ''
    cv = target.CrossValidator(kfold, loss, image_data_generator)

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
