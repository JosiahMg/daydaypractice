"""
功能说明:
使用如下格式的csv文件构造词典


输入数据格式:
csv的数据格式为:
colums = [name:Text , country: Text]

输出数据格式:
x_data:
 - name2id()
 - name_size
y_data:
 - country2id
 - country_size

用途:
文本分类任务

"""
from collections import Counter
import logging
import numpy as np
import config
import pandas as pd


class NameVocab:
    """
    构建字典:
    train_data.keys(): ['name', 'country']
    """
    def __init__(self):
        self.train_data_path = config.name_train_path
        self._id2country = []

        self.build_vocab()

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._country2id = reverse(self._id2country)

        logging.info("Build vocab: words %d, labels %d." % (self.name_size, self.country_size))

    def build_vocab(self):
        data = pd.read_csv(self.train_data_path, encoding='utf-8', header=[0])
        data.columns = ['name', 'country']
        country_counter = Counter(data['country'])

        for country, _ in country_counter.most_common():
            self._id2country.append(country)

    @staticmethod
    def name2id(xs):
        if isinstance(xs, (list, np.ndarray)):
            return [[ord(c) for c in x] for x in xs]
        return [ord(c) for c in xs]

    def country2id(self, xs):
        if isinstance(xs, (list, np.ndarray)):
            return [self._country2id.get(x) for x in xs]
        return self._country2id.get(xs)

    @property
    def name_size(self):
        # 返回 ASCII 码最大值
        return 128

    @property
    def country_size(self):
        return len(self._id2country)


if __name__ == '__main__':
    vocab = NameVocab()
    print(vocab.country_size)
    print(vocab.name_size)


