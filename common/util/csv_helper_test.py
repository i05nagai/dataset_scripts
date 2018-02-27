from . import csv_helper as target


def read_as_dict_test(faker, mocker):
    # mock
    read_data = """
header1, header2,
val1, val2
1, 2
"""
    mock = mocker.mock_open(read_data=read_data)
    mocker.patch.object(target, 'open', mock)
    # call
    path = faker.pystr()
    generator = target.read_as_dict(path)

    expect = [
        {
            'header1': 'val1',
            'header2': 'val2',
        },
        {
            'header1': '1',
            'header1': '2',
        },
    ]

    for i, row in enumerate(generator):
        assert row == expect[i]
