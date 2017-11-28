import requests
import os
import time

from . import filesystem


def _gen_filepath(url, dirpath):
    name = url.split('/')[-1]
    return os.path.join(dirpath, name)


def download_img(url, dirpath):
    r = requests.get(url)
    if r.status_code == 200:
        dirpath = os.path.dirname(dirpath)
        filesystem.make_directory(dirpath)
        filepath = _gen_filepath(url, dirpath)
        with open(filepath, 'wb') as f:
            f.write(r.content)


def download_imgs(urls, dirpath, sleep_time=0.5):
    for url in urls:
        print('  downloading .. {0}'.format(url))
        download_img(url, dirpath)
        time.sleep(sleep_time)
