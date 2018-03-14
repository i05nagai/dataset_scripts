import neologd_normalizer


class NeologdNormalizerTest:

    # before all tests starts
    @classmethod
    def setup_class(cls):
        pass

    # after all tests finish
    @classmethod
    def teardown_class(cls):
        pass

    # before each test start
    def setup(self):
        self.target = neologd_normalizer.NeologdNormalizer()

    # after each test finish
    def teardown(self):
        pass

    def preprocess_test(self):
        # test cases are from
        # https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
        actual = self.target.normalize_neologd("０１２３４５６７８９")
        assert "0123456789" == actual
        actual = self.target.normalize_neologd("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ")
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" == actual
        actual = self.target.normalize_neologd("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ")
        assert "abcdefghijklmnopqrstuvwxyz" == actual
        actual = self.target.normalize_neologd("！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝")
        assert "!\"#$%&'()*+,-./:;<>?@[¥]^_`{|}" == actual
        actual = self.target.normalize_neologd("＝。、・「」")
        assert "＝。、・「」" == actual
        actual = self.target.normalize_neologd("ﾊﾝｶｸ")
        assert "ハンカク" == actual
        actual = self.target.normalize_neologd("o₋o")
        assert "o-o" == actual
        actual = self.target.normalize_neologd("majika━")
        assert "majikaー" == actual
        actual = self.target.normalize_neologd("わ〰い")
        assert "わい" == actual
        actual = self.target.normalize_neologd("スーパーーーー")
        assert "スーパー" == actual
        actual = self.target.normalize_neologd("!#")
        assert "!#" == actual
        actual = self.target.normalize_neologd("ゼンカク　スペース")
        assert "ゼンカクスペース" == actual
        actual = self.target.normalize_neologd("お             お")
        assert "おお" == actual
        actual = self.target.normalize_neologd("      おお")
        assert "おお" == actual
        actual = self.target.normalize_neologd("おお      ")
        assert "おお" == actual
        actual = self.target.normalize_neologd("検索 エンジン 自作 入門 を 買い ました!!!")
        assert "検索エンジン自作入門を買いました!!!" == actual
        actual = self.target.normalize_neologd("アルゴリズム C")
        assert "アルゴリズムC" == actual
        actual = self.target.normalize_neologd("　　　ＰＲＭＬ　　副　読　本　　　")
        assert "PRML副読本" == actual
        actual = self.target.normalize_neologd("Coding the Matrix")
        assert "Coding the Matrix" == actual
        actual = self.target.normalize_neologd("南アルプスの　天然水　Ｓｐａｒｋｉｎｇ　Ｌｅｍｏｎ　レモン一絞り")
        assert "南アルプスの天然水Sparking Lemonレモン一絞り" == actual
        actual = self.target.normalize_neologd("南アルプスの　天然水-　Ｓｐａｒｋｉｎｇ*　Ｌｅｍｏｎ+　レモン一絞り")
        assert "南アルプスの天然水-Sparking*Lemon+レモン一絞り" == actual
