import argparse
import errno
import os
import requests
import time
import urllib.request as request
import random


def get_hyponym(wnid):
    base_url = 'http://www.image-net.org/api'
    url = '{0}/text/wordnet.structure.hyponym?wnid={1}&full=1'
    with request.urlopen(url.format(base_url, wnid)) as req:
        wnids = req.read().decode('utf8').split('\r\n')[1:-1]
        return [hyponym.replace(r'-', '') for hyponym in wnids]


def get_urls(wnid):
    base_url = 'http://www.image-net.org/api'
    url = '{0}/text/imagenet.synset.geturls?wnid={1}'
    with request.urlopen(url.format(base_url, wnid)) as req:
        urls = req.read().decode('utf8').split('\r\n')[0:-1]
        return urls


def get_mapping_between_word_and_wnid():
    url = 'http://image-net.org/archive/words.txt'
    filename = 'words.txt'
    # if filenot exists, download data and save it
    if not os.path.isfile(filename):
        with request.urlopen(url) as req:
            lines = req.read().decode('utf8').split(r'\n')[0:-1]
            with open(filename, 'w') as f:
                f.writelins(lines)

    with open(filename, 'r') as f:
        mapping = {}
        for line in f.readlines():
            k, v = line.split('\t')
            mapping[k] = v.rstrip()
        return mapping


def get_synet_list():
    url = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'
    filename = 'imagenet.synset.obtain_synset_list'
    # if file exists, download data and save it
    if not os.path.isfile(filename):
        with request.urlopen(url) as req:
            lines = req.read().decode('utf8')
            with open(filename, 'w') as f:
                f.write(lines)

    with open(filename, 'r') as f:
        return map(lambda line: line.rstrip(), f.readlines())


def make_directory(path_to_directory):
    try:
        os.makedirs(path_to_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_wnid_list(path_to_wnid_list):
    try:
        with open(path_to_wnid_list, encoding='utf-8') as f:
            files = [line.rstrip() for line in f.readlines()]
    except IOError as e:
        print(e)
    return files


def save_url(url, path, timeout=10):
    try:
        response = requests.get(url, allow_redirects=False, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(e)
        return
    if response.status_code != 200:
        return

    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        return

    data = response.content
    try:
        with open(path, 'wb') as f:
            f.write(data)
    except IOError as e:
        print(e)


def get_num_format(num_urls, index):
    num_digits = len(str(num_urls))
    num_format = '{{0:0{0}d}}'.format(num_digits)
    return num_format.format(index)


def get_images(wnid, data_dir, mapping):
    word = mapping[wnid]
    basepath = os.path.join(data_dir, '{0}_{1}'.format(wnid, word))
    make_directory(basepath)

    urls = get_urls(wnid)
    num_urls = len(urls)
    print('{0} files will be downloaded ...'.format(num_urls))
    for index, url in enumerate(urls):
        extension = url.split('.')[-1]
        filename = '{0}.{1}'.format(get_num_format(num_urls, index), extension)
        filepath = os.path.join(basepath, filename)
        print(' Saving {0}'.format(url))
        save_url(url, filepath)
        time.sleep(0.5 + random.random())


def get_images_recursively(wnid, data_dir, mapping):
    wnid_children = get_hyponym(wnid)
    data_dir_appended = os.path.join(data_dir, wnid)
    for wnid_child in wnid_children:
        get_images(wnid, data_dir, mapping)
        get_images_recursively(wnid_child, data_dir_appended, mapping)


def get(args):
    data_dir = args.data_dir
    wnid = args.wnid
    recursive = args.recursive

    mapping = get_mapping_between_word_and_wnid()
    if recursive:
        get_images_recursively(wnid, data_dir, mapping)
    else:
        get_images(wnid, data_dir, mapping)


def make_argparser():
    # create the top-level parser
    parser = argparse.ArgumentParser(
        prog='PROG',
        description='')
    # subpersers
    subparsers = parser.add_subparsers(
        help='sub-command help')
    # create the subcommand parser
    subparser_get = subparsers.add_parser(
        'get',
        help='help')
    subparser_get.add_argument(
        'wnid',
        type=str,
        help='wnid which downalod')
    subparser_get.add_argument(
        '--data_dir',
        type=str,
        help='download directory',
        default='images')
    subparser_get.add_argument(
        '--max',
        type=int,
        help='maximum number of images',
        default=10)
    subparser_get.add_argument(
        '--recursive',
        action='store_true',
        help='maximum number of images')
    subparser_get.set_defaults(func=get)

    return parser


def main():
    parser = make_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
