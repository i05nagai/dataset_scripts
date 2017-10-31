import os
import random


def move_randomly(path_from, path_to, num):
    files = os.listdir(path_from)
    files = random.sample(files, num)
    for fname in files:
        from_abspath = os.path.join(path_from, fname)
        to_abspath = os.path.join(path_to, fname)
        os.rename(from_abspath, to_abspath)
        print('{0} -> {1}'.format(from_abspath, to_abspath))
