import requests
import os
import time

from . import url as util_url
from . import filesystem


def _gen_filepath(url, dirpath):
    name = url.split('/')[-1]
    return os.path.join(dirpath, name)


def _gen_filepath_from_url(url, dirpath):
    name = util_url.url_to_filename(url)
    return os.path.join(dirpath, name)


def download_img(url, dirpath, filename_from_url=False):
    r = requests.get(url)
    if r.status_code == 200:
        dirpath = os.path.dirname(dirpath)
        filesystem.make_directory(dirpath)
        if filename_from_url:
            filepath = _gen_filepath_from_url(url, dirpath)
        else:
            filepath = _gen_filepath(url, dirpath)
        with open(filepath, 'wb') as f:
            f.write(r.content)


def download_imgs(urls, dirpath, sleep_time=0.5, filename_from_url=False):
    for url in urls:
        print('  downloading .. {0}'.format(url))
        download_img(url, dirpath, filename_from_url)
        time.sleep(sleep_time)
