# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import bigquery


def get_data(
        project_id,
        path_to_query,
        path_to_output):
    client = bigquery.get_client(project_id)
    query_runner = bigquery.QueryRunner(client)
    result = query_runner.run_from_file(path_to_query)

    return result


def to_double_array(data, with_header=True):
    row_dict = data[0]
    keys = [key for key in row_dict.keys()]
    if with_header:
        yield keys
    yield [row_dict[key] for key in keys]
    for row_dict in data:
        yield [row_dict[key] for key in keys]
