from PIL import Image
import argparse
import os
import re


def delete_invalid_images(basepath):
    count = 0
    for root, subdirs, files in os.walk(basepath):
        for file in files:
            path_to_file = os.path.join(root, file)
            if delete_invalid_image(path_to_file):
                count += 1
    return count


def delete_invalid_image(path):
    if not is_valid_image(path) and not is_valid_image_type(path):
        print('Deleting ...{0}'.format(path))
        delete_file(path)
        return True
    return False


def delete_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def is_valid_image(path):
    # Try to open image with tools of Pillow
    try:
        image = Image.open(path)
        image.verify()
        image.close()
    except Exception as e:
        print(e)
        return False

    return True


def is_valid_image_type(path, valid_type=['jpeg', 'png']):
    try:
        image = Image.open(path)
        if str(image.format).lower() in valid_type:
            return True
        image.close()
    except Exception as e:
        print(e)
        return False

    return False


def infer_extension(path):
    try:
        image = Image.open(path)
        extension = str(image.format).lower()
        image.close()
        # if extension is not correct
    except Exception as e:
        print(e)
        return ''
    mapping = {
        'jpeg': 'jpg',
    }
    if extension in mapping:
        return mapping[extension]
    else:
        return extension


def change_extension(path, extension_new):
    basepath, extension = os.path.splitext(path)
    path_new = '{0}.{1}'.format(basepath, extension_new)
    os.rename(path, path_new)


def set_appropriate_extension(basepath):
    count = 0
    for root, subdirs, files in os.walk(basepath):
        for file in files:
            path_to_file = os.path.join(root, file)
            extension_new = infer_extension(path_to_file)
            extension_with_period = os.path.splitext(path_to_file)[1]
            extension = re.sub(r'\A\.', '', extension_with_period)
            if extension_new == '':
                print('  cannot infer the extension of {0}'.format(
                    path_to_file))
            elif extension_new != extension:
                count += 1
                print('  {0}: {1} -> {2}'.format(
                    path_to_file, extension, extension_new))
                change_extension(path_to_file, extension_new)
    return count


def make_argparser():
    # create the top-level parser
    parser = argparse.ArgumentParser(
        prog='PROG',
        description='image')
    # subpersers
    subparsers = parser.add_subparsers(
        help='rm')
    # create the subcommand parser
    subparser_rm = subparsers.add_parser(
        'rm',
        help='remove invalid images')
    subparser_rm.add_argument(
        "paths",
        nargs="+",
        help="paths to file")
    subparser_rm.set_defaults(func=rm)
    # subcommand parser for ext
    subparser_ext = subparsers.add_parser(
        'ext',
        help='set appropriate exntension')
    subparser_ext.add_argument(
        "paths",
        nargs="+",
        help="paths to file")
    subparser_ext.set_defaults(func=ext)

    return parser


def rm(argparse):
    """rm

    :param argparse:

    Usage
    ======
    python thisfile.py rm path/to/dir1 path/to/dir2
    """
    paths = argparse.paths
    for path in paths:
        print('Remove invalid files in {0}'.format(path))
        count = delete_invalid_images(path)
        print('  {0} files deleted'.format(count))


def ext(argparse):
    """ext

    :param argparse:

    Usage
    ======
    python thisfile.py ext path/to/dir1 path/to/dir2
    """
    paths = argparse.paths
    for path in paths:
        print('Rename files in {0}'.format(path))
        count = set_appropriate_extension(path)
        print('  {0} files renamed'.format(count))


def main():
    parser = make_argparser()
    argparse = parser.parse_args()
    argparse.func(argparse)


if __name__ == '__main__':
    main()
