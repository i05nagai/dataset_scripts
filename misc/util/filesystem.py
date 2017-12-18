from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import errno
import os
import json
import shutil


def get_filename(path_to_dir, recursive=True):
    if not recursive:
        # return only file name
        return os.listdir(path_to_dir)
    else:
        filenames = []
        for root, subdirs, files in os.walk(path_to_dir):
            for fname in files:
                filenames.append(fname)
        return filenames


def get_filepath(path_to_dir, recursive=True):
    if not recursive:
        filenames = os.listdir(path_to_dir)
        filepaths = [os.path.join(path_to_dir, name) for name in filenames]
        return filepaths
    else:
        filepaths = []
        for root, subdirs, files in os.walk(path_to_dir):
            for fnames in files:
                path_to_file = os.path.join(root, fnames)
                filepaths.append(path_to_file)
        return filepaths


def get_parent_dir(path_to_dir):
    return os.path.dirname(path_to_dir)


def add_prefix_to_filename(path, prefix, separator='_'):
    """add_prefix_to_filename
    Add prefix to filename

    :param path:
    :param prefix:
    :param separator:
    """
    filename = os.path.basename(path)
    filename_new = '{0}{1}{2}'.format(prefix, separator, filename)
    dirpath = os.path.dirname(path)
    path_new = os.path.join(dirpath, filename_new)
    return path_new


def add_suffix_to_filename(path: str, suffix: str, separator='_') -> str:
    """add_suffix_to_filename
    Add suffix to filename

    :param path:
    :param suffix:
    :param separator:

    :return: suffix with filename
    :rtype: str
    """
    filename, ext = os.path.splitext(os.path.basename(path))
    filename_new = '{0}{1}{2}{3}'.format(filename, separator, suffix, ext)
    dirpath = os.path.dirname(path)
    path_new = os.path.join(dirpath, filename_new)
    return path_new


def rename_filename(path: str, filename: str):
    basepath = os.path.basename(path)
    path_new = '{0}/new_filename'.format(basepath)
    rename_path(path, path_new)


def rename_path(from_path: str, to_path: str):
    os.rename(from_path, to_path)


def rename_filename_with_suffix(
        path_to_dir: str, suffix: str, separator='_') -> str:
    """rename_filename_with_suffix

    :param path_to_dir:
    :param suffix:
    :param separator:

    :rtype: str
    """
    paths = get_filepath(path_to_dir)
    for path in paths:
        path_new = add_suffix_to_filename(path, suffix, separator)
        rename_path(path, path_new)


def make_directory(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_as_json(path, json_dict):
    with open(path, 'w') as f:
        json.dump(json_dict, f, indent=2, sort_keys=True)


def get_filepath_with_condition(path_to_dir, functor):
    path_matched = []
    for path in get_filepath(path_to_dir):
        if functor(path):
            path_matched.append(path)
    return path_matched


def change_directory_in_path(path: str, path_to_dir: str):
    filename = os.path.basename(path)
    return os.path.join(path_to_dir, filename)


def copy_file(path_from: str, path_to: str):
    shutil.copyfile(path_from, path_to)


def copy_file_with_condition(
        path_to_dir: str,
        functor,
        path_to_copy_dir: str,
        offset: int=0):
    path_matched = []
    paths = get_filepath(path_to_dir)
    for i, path in enumerate(paths[offset:], offset):
        print('{0}-{1}'.format(i, path))
        if functor(path):
            path_matched.append(path)
            path_new = change_directory_in_path(path, path_to_copy_dir)
            copy_file(path, path_new)

    return path_matched


def classify_file_with_condition(
        path_to_dir: str,
        functor,
        path_to_copy_dirs: str,
        offset: int=0) -> None:
    paths = get_filepath(path_to_dir)
    for i, path in enumerate(paths[offset:], offset):
        print('{0}-{1}'.format(i, path))
        index_category = functor(path)
        path_to_copy_dir = path_to_copy_dirs[index_category]
        path_new = change_directory_in_path(path, path_to_copy_dir)
        copy_file(path, path_new)
