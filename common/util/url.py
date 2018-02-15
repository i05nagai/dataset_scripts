# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os


def url_to_filename(url):
    url = url.replace('://', '___')
    url = url.replace('/', '_')
    return url


def paths_to_uri(paths):
    scheme = 'file://'
    paths = [scheme + os.path.abspath(path) for path in paths]
    return paths


def main():
    url = ''
    filename = url_to_filename(url)
    print(filename, end='')


if __name__ == '__main__':
    main()
