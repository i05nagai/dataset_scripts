import csv


def read_as_dict(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # skip header
        header = next(reader)
        for row in reader:
            yield dict(zip(header, row))
