from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import image_net_util


def main():
    """main

    Example
    ========
    python wordnet.py --path imagenet/images/full --wordfile words.txt

    """
    parser = argparse.ArgumentParser(description="description of program")
    parser.add_argument("--path",
                        type=str,
                        help="path to dir")
    parser.add_argument("--wordfile",
                        type=str,
                        help="path to words.txt dir")
    args = parser.parse_args()
    path = args.path
    path_to_word = args.wordfile

    # read file
    mapping = image_net_util.read_word_mapping(path_to_word)

    files = os.listdir(path)
    not_found = {}
    for wnid in files:
        if wnid in mapping:
            print('{0}\t{1}'.format(wnid, mapping[wnid]))
        else:
            not_found[wnid] = 1
    for wnid in not_found:
        print(' no mapping found: {0}'.format(wnid))


if __name__ == '__main__':
    main()
