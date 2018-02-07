import os


def _join_abspath(*args):
    return os.path.abspath(os.path.join(*args))


PATH_TO_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_TO_PROJECT_DIR = _join_abspath(PATH_TO_THIS_DIR, '..')
PATH_TO_CREDENTIALS = _join_abspath(PATH_TO_PROJECT_DIR, 'credential')
PATH_TO_DATA = _join_abspath(PATH_TO_PROJECT_DIR, 'data')
