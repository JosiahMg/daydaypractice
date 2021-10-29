import config
import jieba
from typing import Text


class JiebaTokenizer:
    stopword_path = config.stopword_path
    userdict_path = config.userdict_path

    def __init__(self):
        self.stopwords = set()
        self.read_in_stopword()

    def load_userdict(self):
        jieba.load_userdict(self.userdict_path)

    def read_in_stopword(self):
        file_obj = open(self.stopword_path, mode='r', encoding='utf-8')
        while True:
            line = file_obj.readline()
            line = line.strip('\r\n')
            if not line:
                break
            self.stopwords.add(line)
        file_obj.close()

    def tokenize(self, sentence: Text, stopword=True, userdict=True, lower=True, cut_all=False):
        if userdict:
            self.load_userdict()
        if lower:
            sentence = sentence.lower()
        seg_list = jieba.cut(sentence, cut_all)
        results = []
        for seg in seg_list:
            if stopword and seg in self.stopwords:
                continue
            results.append(seg)
        return results


if __name__ == '__main__':
    data = 'python常见数据结构有哪些'
    print(JiebaTokenizer().tokenize(data))
