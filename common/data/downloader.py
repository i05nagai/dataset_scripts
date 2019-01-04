from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    import urllib.request as request
except Exception:
    import urllib2 as request


def download_cifar_10(data_dir):
    CIFAR_10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_file(CIFAR_10_URL, data_dir)


def retrieve_file(url):
    response = request.urlopen(url)
    return response.read().decode('utf-8')


def download_file(url, filename):
    request.urlretrieve(url, filename)
