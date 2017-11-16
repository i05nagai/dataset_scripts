import os
import os.path
import settings
import sklearn.model_selection as model_selection
import numpy as np


def _get_files(path):
    file_paths = []
    for root, subdirs, files in os.walk(path):
        for fname in files:
            path_to_file = os.path.join(root, fname)
            file_paths.append(path_to_file)
    return file_paths


def get_labels(paths, classes):
    labels = []
    for path in paths:
        dir_name = os.path.basename(os.path.dirname(path))
        if dir_name in classes:
            index = classes.index(dir_name)
            labels.append(index)
        else:
            raise ValueError('directory name must be same as one of classes')
    return labels


def get_paths_and_labels(path_to_train, path_to_validation, classes):
    paths = []
    # train file path
    paths = _get_files(path_to_train)
    # validation file path
    paths += _get_files(path_to_validation)

    labels = get_labels(paths, classes)
    return paths, labels


def cross_validation(
        model, path_to_train, path_to_validation, classes, n_splits):
    xs, ys = get_paths_and_labels(path_to_train, path_to_validation, classes)
    xs = np.array(xs)
    ys = np.array(ys)
    kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train, test in kfold.split(xs, ys):
        print(xs[train])
        print(ys[train])
        # result = model.predict(xs[train])
        # max_index = np.argmax(result)


def main():
    path_to_train = settings.path_to_image_train
    path_to_validation = settings.path_to_image_validation
    classes = settings.categories

    model = None
    cross_validation(
        model, path_to_train, path_to_validation, classes, n_splits=2)


if __name__ == '__main__':
    main()
