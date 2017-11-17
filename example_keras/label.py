from . import category
import copy
import json
import os
import random
import shutil
import util


class LabelMaker(object):

    TRAIN_DIR_NAME = 'train'
    VALIDATION_DIR_NAME = 'validation'

    def __init__(self,
                 path_to_output,
                 validation_rate,
                 validation_min,
                 max_num_image_per_class):
        self.path_to_output = path_to_output
        self.validation_rate = validation_rate
        self.validation_min = validation_min
        self.validation_count = 0
        self.train_count = 0
        self.max_num_image_per_class = max_num_image_per_class

    def _get_files(self, path):
        return [os.path.join(path, fname) for fname in os.listdir(path)]

    def _copy_files(self, paths, path_to_dir, prefix, count):
        # restrict to max_num_image_per_class
        if len(paths) > self.max_num_image_per_class:
            paths = paths

        util.make_directory(path_to_dir)
        for path in paths:
            name = os.path.basename(path)
            filename = '{0:06d}_{1}_{2}'.format(count, prefix, name)
            path_to_output = os.path.join(path_to_dir, filename)
            shutil.copy2(path, path_to_output)
            print('{0} --> {1}'.format(path, path_to_output))
            count += 1
        return count

    def _get_num_train_and_num_validation(self, num_images):
        # use up to max_num_image_per_class
        if num_images > self.max_num_image_per_class:
            num_images = self.max_num_image_per_class
        num_validation = int(num_images * self.validation_rate)
        # ensure the minimum value of
        if num_validation < self.validation_min:
            num_validation = self.validation_min
        num_train = num_images - num_validation
        return num_train, num_validation

    def make_train_and_validation_label(self, label_info):
        files = self._get_files(label_info['path'])
        samples = random.sample(files, len(files))

        num_train, num_validation = self._get_num_train_and_num_validation(
            len(files))
        num_images = num_train + num_validation
        print(num_train, num_validation)
        paths_train = samples[0:num_train]
        paths_validation = samples[num_train:num_images]
        # copy train
        path_to_train = os.path.join(
            self.path_to_output, self.TRAIN_DIR_NAME, label_info['label'])
        self.train_count = self._copy_files(
            paths_train, path_to_train, label_info['prefix'], self.train_count)
        label_info['num_train'] = num_train
        # copy validation
        path_to_validation = os.path.join(
            self.path_to_output, self.VALIDATION_DIR_NAME, label_info['label'])
        self.validation_count = self._copy_files(
            paths_validation,
            path_to_validation,
            label_info['prefix'],
            self.validation_count)
        label_info['num_validation'] = num_validation

    def make_dataset(self, label_infos):
        label_infos = copy.deepcopy(label_infos)
        for label_info in label_infos:
            self.make_train_and_validation_label(label_info)
        return label_infos

    def save_labels(self, label_infos, path='label.json'):
        # train labels
        labels_train = []
        for i, label_info in enumerate(label_infos):
            labels_train += [i] * label_info['num_train']
        # validation labels
        labels_validation = []
        for i, label_info in enumerate(label_infos):
            labels_validation += [i] * label_info['num_validation']
        # write to files
        output = {
            'labels_train': labels_train,
            'labels_validation': labels_validation,
        }
        path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
        abspath = os.path.join(path_to_this_dir, path)
        with open(abspath, 'w') as f:
            json.dump(output, f, indent=2, sort_keys=True)


def main():
    path_to_this_dir = os.path.abspath(os.path.dirname(__file__))
    basepath = os.path.join(path_to_this_dir, './image')

    config = category.read_settings()
    label_infos = config['categories']

    for label_info in label_infos:
        label_info['path'] = os.path.join(basepath, label_info['path'])

    validation_rate = 0.05
    validation_min = 2
    max_num_image_per_class = 10
    path_to_output = basepath
    maker = LabelMaker(
        path_to_output,
        validation_rate,
        validation_min,
        max_num_image_per_class)
    label_infos = maker.make_dataset(label_infos)


if __name__ == '__main__':
    main()
