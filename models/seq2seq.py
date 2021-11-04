"""
将字符串"12345"->"123450"
每个字符串后面加一个'0'
步骤:
1. 构造词典
2. 构造dataloader
3. 构造模型
4. 

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pprint import pprint


train_batch_size = 2
seq_max_len = 9
# 1. 构造词典


class NumVocab:
    """此项目无需词典
    """

    def transform(self, inputs, max_len=seq_max_len):
        if len(inputs) < max_len:
            inputs = inputs + [0]*(max_len-len(inputs))
        return inputs


# 2. 准备数据


class NumDataset(Dataset):
    """构造Dataset

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, vocab=None):
        self.data = np.random.randint(0, 1e8, size=[500000])
        self.vocab = vocab

    def __getitem__(self, index: int):
        inputs = list(map(int, str(self.data[index])))
        labels = inputs+[0]
        if self.vocab:
            inputs = self.vocab.transform(inputs)
            labels = self.vocab.transform(labels)
        return np.array(inputs), np.array(labels)

    def __len__(self) -> int:
        return self.data.shape[0]


num_dataset = NumDataset(vocab=NumVocab())
print(num_dataset[0])

# 构造DataLoader
train_data_loader = DataLoader(
    num_dataset, batch_size=train_batch_size, shuffle=True)


if __name__ == '__main__':
    for inputs, labels in train_data_loader:
        print(inputs)
        print(labels)
        break
