# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def url_to_filename(url):

    url = url.replace('://', '___')
    url = url.replace('/', '_')
    return url


def main():
    url = ''
    filename = url_to_filename(url)
    print(filename, end='')


if __name__ == '__main__':
    main()
