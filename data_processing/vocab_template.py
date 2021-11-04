"""
功能说明:
构造词典的模板程序
    1. 对所有句子进行分词
    2. 词语存入字典，根据次数对词语进行过滤，并统计次数
    3. 实现文本转换数字序列的方法
    4. 实现数字序列转文本方法


输入数据格式:
List[List[Text]]

输出数据格式:
1. word -> index
2. index -> word

用途:
数据集准备的前提

"""
import pickle
from typing import List, Text
import numpy as np
import config
import os
from tqdm import tqdm


class Word2Sequence:
    """
    1. 对所有句子进行分词
    2. 词语存入字典，根据次数对词语进行过滤，并统计次数
    3. 实现文本转换数字序列的方法
    4. 实现数字序列转文本方法
    """
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.fited = False
        self.count = {}

    def to_index(self, word: Text):
        assert self.fited == True  # 必须先进行fit操作
        return self.dict.get(word, self.UNK)

    def to_word(self, index):
        assert self.fited
        return self.inverse_dict.get(index, self.UNK_TAG)

    def __len__(self):
        return len(self.dict)

    def fit(self, sentences: List):
        for sentence in sentences:
            if isinstance(sentence, (list, np.ndarray)):
                for word in sentence:
                    self.count[word] = self.count.get(word, 0) + 1
            else:
                self.count[sentence] = self.count.get(sentence, 0) + 1

    def build_vocab(self, min_count=1, max_count=None, max_features=None):
        # 限制最小词频
        if min_count is not None:
            self.count = {word: value for word, value in self.count.items() if value >= min_count}
        # 限制最大词频
        if max_count is not None:
            self.count = {word: value for word, value in self.count.items() if value <= max_count}
        # 限制词的个数
        if isinstance(max_features, int):
            self.count = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)
            if max_features is not None and len(self.count) > max_features:
                self.count = self.count[:max_features]
            for w, _ in self.count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(self.count.keys()):
                self.dict[w] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        self.fited = True

    def transform(self, sentence: List[Text], max_len: int=None):
        """
        文字转换成数字
        """
        assert self.fited, "fit first"
        if max_len is not None:
            r = [self.PAD]*max_len
        else:
            r = [self.PAD]*len(sentence)
        if max_len is not None and len(sentence)>max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r, dtype=np.int64)

    def inverse_transform(self, indices):
        """
        数字转换成文字
        :param indices:
        :return:
        """
        return [self.to_word(i) for i in indices]


if __name__ == '__main__':
    w2s = Word2Sequence()
    path = config.imdb_train_path
    paths = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for data_path in paths:
        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('txt')]
        for file_path in tqdm(file_paths, desc=str(os.path.basename(data_path))):
            # TODO tokenizer
            sentence = open(file_path, encoding='utf-8').read().split()
            w2s.fit(sentence)

    w2s.build_vocab(min_count=10, max_features=10000)
    pickle.dump(w2s, open(config.imdb_vocab, 'wb'))
    print(len(w2s))

