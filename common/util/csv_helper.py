import csv


def read_as_dict(path, skipHeader=True):
    """read_as_dict

    :param path:
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not skipHeader:
            yield header

        for row in reader:
            yield dict(zip(header, row))


def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


def write_csv(path, data):
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


def write_dict_as_csv(path, data):
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for k, v in data.items():
            writer.writerow([k, v])


def write_array_of_dict(path, data, write_header=True):
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
    :param data:
    """
    with open(path, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
