"""
将 字符串 "0-9" 变成 数字0-9
"""
from typing import Text, List


class NumVocab:
    """此项目无需词典
    """
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD,
                     self.UNK_TAG: self.UNK,
                     self.SOS_TAG: self.SOS,
                     self.EOS_TAG: self.EOS
                     }
        for i in range(10):
            self.dict[str(i)] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence: List[Text], max_len, add_eos=False):
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        seq_len = len(sentence)
        if add_eos:
            sentence = sentence + [self.EOS_TAG]
        if seq_len < max_len:
            sentence = sentence + [self.PAD_TAG]*(max_len-seq_len)
        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def __len__(self):
        return len(self.dict)


vocab = NumVocab()


if __name__ == '__main__':

    print(vocab.dict)
    print(vocab.inverse_dict)
