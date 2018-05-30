import csv
import pandas as pd
import datetime


MINOR_TO_MAJOR = {
    'r': 'procs',
    'b': 'procs',
    'swpd': 'memory',
    'free': 'memory',
    'buff': 'memory',
    'cache': 'memory',
    'inact': 'memory',   # to support the vmstat -a option if required
    'active': 'memory',  # to support the vmstat -a option if required
    'si': 'swap',
    'st': 'swap',
    'so': 'swap',
    'bi': 'io',
    'bo': 'io',
    'in': 'system',
    'cs': 'system',
    'us': 'cpu',
    'sy': 'cpu',
    'id': 'cpu',
    'wa': 'cpu'
}


def read_vmstat(path_csv):
    """read_vmstat
    http://www.eurion.net/python-snippets/snippet/vmstat%20Reader.html
    """
    minors = []
    content = dict([(h, {}) for h in set(MINOR_TO_MAJOR.values())])

    reader = csv.reader(open(path_csv), delimiter=' ', skipinitialspace=True)
    for row in reader:
        if reader.line_num == 1:
            """
            Ignore the first line as it contains major headers.
            """
        elif reader.line_num == 2:
            """
            If we are on the first line, create the headers list from the first row.
            We also keep a copy of the minor headers, in the order that they appear
            in the file to ensure that we can map the values to the correct entry
            in the content map.
            """
            minors = row
            for h in row:
                content[MINOR_TO_MAJOR[h]][h] = []

        elif row[0] != minors[0] and row[0] != MINOR_TO_MAJOR[minors[0]]:
            """
            If the -n option was not specified when running the vmstat command,
            major and minor headers are repeated so we need to ensure that we
            ignore such lines and only deal with lines that contain actual data.
            For each value in the row, we append it to the respective entry in
            the content dictionary. In addition, we transform the value to an int
            before appending it as we know that the content of the log should only
            have integer values.
            """
            for i, v in enumerate(row):
                content[MINOR_TO_MAJOR[minors[i]]][minors[i]].append(int(v))

    return content


def read_vmstat_as_double_array(path_csv):
    """read_as_dataframe
    """
    content = []
    minors = []

    reader = csv.reader(open(path_csv), delimiter=' ', skipinitialspace=True)
    for row in reader:
        if reader.line_num == 1:
            """
            Ignore the first line as it contains major headers.
            """
        elif reader.line_num == 2:
            """
            If we are on the first line, create the headers list from the first row.
            We also keep a copy of the minor headers, in the order that they appear
            in the file to ensure that we can map the values to the correct entry
            in the content map.
            """
            minors = row
            # header
            content.append(row)

        elif row[0] != minors[0] and row[0] != MINOR_TO_MAJOR[minors[0]]:
            """
            If the -n option was not specified when running the vmstat command,
            major and minor headers are repeated so we need to ensure that we
            ignore such lines and only deal with lines that contain actual data.
            For each value in the row, we append it to the respective entry in
            the content dictionary. In addition, we transform the value to an int
            before appending it as we know that the content of the log should only
            have integer values.
            """
            content.append(row)

    return content


def _add_timestamp(df, start_datetime, interval):
    # add timeseriress
    def add_timestamp(row):
        if row.name == 0:
            return start_datetime
        else:
            shift = datetime.timedelta(seconds=int(row.name) * interval)
            return shift + start_datetime

    df['timestamp'] = df.apply(add_timestamp, axis=1)
    return df


def read_vmstat_as_dataframe(path_csv, start_datetime=None, interval=10):
    """read_vmstat_as_dataframe

    :param path_csv:
    :param interval: secs
    """
    double_array = read_vmstat_as_double_array(path_csv)
    df = pd.DataFrame(double_array[1:], columns=double_array[0], dtype=None)
    df = df.apply(pd.to_numeric)

    if start_datetime is not None:
        df = _add_timestamp(df, start_datetime, interval)
    return df
