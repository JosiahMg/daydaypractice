"""
将字符串"12345"->"123450"
每个字符串后面加一个'0'

步骤:
1. 构造词典
2. 构造dataloader

"""
from typing import List, Text, Tuple
from vocabs.vocab_number import vocab
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pprint import pprint


seq_max_len = 9
batch_size = 128

# 1. 构造词典


# 2. 准备数据


class NumDataset(Dataset):
    """构造Dataset

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self):
        np.random.seed(10)
        self.data = np.random.randint(0, 1e8, size=[500000])

    def __getitem__(self, index: int):
        x = list(str(self.data[index]))
        y = x + ['0']
        return x, y, len(x), len(y)

    def __len__(self) -> int:
        return self.data.shape[0]


def collate_fn(batch):
    """
    batch.shape: ([(input, label, input_len, label_len), ...])
    """
    batch = sorted(batch, key=lambda x: x[3], reverse=True)
    inputs, labels, input_len, label_len = list(zip(*batch))

    inputs = torch.LongTensor([vocab.transform(i, max_len=seq_max_len, add_eos=False)
                               for i in inputs])
    labels = torch.LongTensor([vocab.transform(i, max_len=seq_max_len, add_eos=True)
                               for i in labels])

    input_len = [x if x < seq_max_len else seq_max_len for x in input_len]
    label_len = [x if x < seq_max_len else seq_max_len for x in label_len]

    input_len = torch.LongTensor(input_len)
    label_len = torch.LongTensor(label_len)

    return inputs, labels, input_len, label_len


num_dataset = NumDataset()
# 构造DataLoader
train_data_loader = DataLoader(
    num_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    for inputs, labels, input_len, target_len in train_data_loader:
        print('inputs:', inputs.shape)
        print('labels: ', labels.shape)
        print('inputs len: ', input_len)
        print('target_len: ', target_len)
        break
