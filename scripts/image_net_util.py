def read_word_mapping(path):
    try:
        with open(path, 'r') as f:
            lines = [line.split('\t') for line in f.readlines()]
            mapping = {k: v.rstrip() for k, v in lines}
    except IOError as e:
        print(e)
    return mapping
