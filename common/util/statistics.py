import collections
import numpy as np


def calculate_statistics(
        data,
        label_map,
        value_map):
    """calculate_statistics

    summation
    average
    median
    count

    :param data:
    :type data: list of list
    :param label_map:
    :type label_map: Callable
    :param value_map:
    :type value_map: Callable
    """
    # read as double list
    group = collections.defaultdict(list)
    for row in data:
        label = label_map(row)
        value = value_map(row)
        group[label].append(value)

    # calculate statistics
    statistics = {}
    for key, values in group.items():
        summation = sum(values)
        average = np.mean(values)
        median = np.median(values)
        count = len(values)
        statistics[key] = {
            'summation': summation,
            'average': average,
            'median': median,
            'count': count,
        }

    return statistics
