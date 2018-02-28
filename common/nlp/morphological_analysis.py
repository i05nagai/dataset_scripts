import MeCab

import os
import urllib


def get_stop_words():
    """get_stop_words
    from slothlib
    """
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    lines = urllib.request.urlopen(url)
    stopwords = [line.decode('utf-8').strip() for line in lines]
    stopwords = [ss for ss in stopwords if not ss == '']
    return stopwords


class Parser(object):

    def __init__(self, paths_to_dict=['mecab-ipadic-neologd']):
        self.mecab_base_dir = '/usr/lib/mecab/dic'
        # create options
        dict_dirs = []
        for path_relative in paths_to_dict:
            path = os.path.join(self.mecab_base_dir, path_relative)
            dict_dirs.append(path)
        options = map(lambda x: '--dicdir={0}'.format(x), dict_dirs)
        options_str = ' '.join(options)
        self.mecab = MeCab.Tagger(options_str)

        # warm up
        # this is for fixing bug
        # see https://qiita.com/piruty/items/ce218090eae53b775b79
        self.mecab.parse('')

    def parse(self, text):
        """ break string up into tokens and stem words """
        m = self.mecab.parseToNode(text)
        res = []
        while m:
            feature = self._process_feature(m.feature)
            res.append([m.surface] + feature)
            m = m.next
        return res

    def _process_feature(self, feature):
        feature = feature.split(',')
        # 品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
        feature = list(map(lambda x: x.replace('*', ''), feature))
        # reduce size and processing time
        feature[0] = feature[0].replace('BOS/EOS', '0')
        return feature
