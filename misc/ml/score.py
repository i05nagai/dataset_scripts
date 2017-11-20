import csv


def get_top_n_prediction_from_file(path, top_n=1):
    """get_top_n_prediction_from_file

    Contents of file
    ================
    Contents must be csv.

        id,label1,label2,label3
        id_num,score1,score2,score3

    :param path:
    :param top_n:
    """
    outputs = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            uri = row[0]
            row_pair = list(zip(header[1:], row[1:]))
            row_pair_sorted = sorted(row_pair, key=lambda x: x[1], reverse=True)
            row_pair_top_n = row_pair_sorted[0:top_n]
            result = {k: v for k, v in row_pair_top_n}
            result['uri'] = uri
            outputs.append(result)
    return outputs
