

def _read_template(path='template.html'):
    try:
        with open(path, 'r') as f:
            data = f.read()
    except IOError as e:
        print(e)
    return data


def get_html(body):
    data = _read_template()
    return data.format(body)
