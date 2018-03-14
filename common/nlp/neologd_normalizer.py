# -*- coding: utf-8 -*-
"""
This script is based on
https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
with modifications.
"""
import re
import unicodedata


class NeologdNormalizer(object):

    def __init__(self):
        self.translation_table = self._maketrans(
            '!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
            '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」')
        print(self.translation_table)

    def _maketrans(self, f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    def normalize_neologd(self, s):
        s = s.strip()
        s = self.unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
        s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
        s = s.translate(self.translation_table)

        s = remove_extra_spaces(s)
        # keep ＝,・,「,」
        s = self.unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        return s

    def _norm(self, c, pt):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    def unicode_normalize(self, cls, s):
        pt = re.compile('([{}]+)'.format(cls))

        data = list((self._norm(x, pt) for x in re.split(pt, s)))
        s = ''.join(data)
        s = re.sub('－', '-', s)
        return s


def remove_space_between(cls1, cls2, s):
    p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
    while p.search(s):
        s = p.sub(r'\1\2', s)
    return s


def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s
