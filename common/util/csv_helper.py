# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import csv


def read_as_dict(path, skipHeader=True):
    """read_as_dict

    :param path:
    :type path: str
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not skipHeader:
            yield header

        for row in reader:
            yield dict(zip(header, row))


def read_tsv(path):
    """read_tsv

    :param path:
    :type path: str
    """
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            yield row


def read_csv(path):
    """read_csv

    :param path:
    :type path: str
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


def write_csv(path, data):
    """write_csv

    :param path:
    :type path: str
    :param data:
    :type data: list of list
    """
    with open(path, 'w') as f:
        writer = csv.writer(
            f,
            lineterminator='\n',
            quotechar='"',
            quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)


def write_dict_as_csv(path, data):
    """write_dict_as_csv

    :param path:
    :param data:
    :type data: dict

    Examples
    ========
    >>> path = '/path/to/output'
    >>> data = {
    >>>     'value11': 'value12',
    >>>     'value21': 'value22',
    >>> }
    >>> write_dict_as_csv(path, data)
    >>> # generated csv is formed as follows:
    >>> # valu11,value12
    >>> # value21,value22
    """
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for k, v in data.items():
            writer.writerow([k, v])


def write_array_of_dict(path, data, write_header=True):
    """write_array_of_dict

    :param path:
    :type path: str
    :param data:
    :type data: list of dict
    :param write_header:
    """
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        # write header
        if write_header:
            header = data[0]
            keys = [key for key in header.keys()]
            writer.writerow(keys)

        # write data
        for row_dict in data:
            values = [row_dict[key] for key in keys]
            writer.writerow(values)


def append_csv(path, data):
    """append_csv

    :param path:
    :type path: str
    :param data:
    :list data:
    """
    with open(path, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
