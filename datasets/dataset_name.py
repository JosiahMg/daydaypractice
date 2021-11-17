"""
1. dataset返回的每个样本的长度没有统一，返回list类型
2. 定义collate_fn函数自定义dataloader
3. 每个batch数据使用make_tensors函数统一长度并转换为tensor
"""
import torch
from torch.utils.data import DataLoader, Dataset
import config
import pandas as pd
import numpy as np
from data_processing import vocab_name


class NameDataset(Dataset):
    def __init__(self, vocab=None, is_train=True):
        filename = config.name_train_path if is_train else config.name_test_path
        df = pd.read_csv(filename, sep=',', header=None, encoding='utf-8')
        df.columns = ['names', 'countries']
        self.names = df['names'].to_numpy()
        self.countries = df['countries'].to_numpy()
        self.vocab = vocab

    def __getitem__(self, index: int):
        name = self.vocab.name2id(
            self.names[index]) if self.vocab else self.names[index]
        country = self.vocab.country2id(
            self.countries[index]) if self.vocab else self.countries[index]
        return name, country

    def __len__(self) -> int:
        return len(self.names)

    def get_country_num(self):
        return len(set(self.countries))


def collate_fn(batch):
    return list(zip(*batch))


def make_tensors(names, countries):
    seq_lens = torch.LongTensor([len(name) for name in names])

    countries = torch.LongTensor(countries)

    seq_tensor = torch.zeros(len(names), seq_lens.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(names, seq_lens), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lens, perm_idx = seq_lens.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return seq_tensor, seq_lens, countries


if __name__ == '__main__':

    trainset = NameDataset(vocab=vocab_name.NameVocab(), is_train=True)
    trainloader = DataLoader(trainset, batch_size=2,
                             shuffle=True, collate_fn=collate_fn)

    testset = NameDataset(vocab=vocab_name.NameVocab(), is_train=False)
    testloader = DataLoader(testset, batch_size=2,
                            shuffle=True, collate_fn=collate_fn)
    for names, countries in testloader:
        print('testloader name: ', names)
        print('testloader country: ', countries)
        break

    for names, countries in trainloader:
        names, lens, countries = make_tensors(names, countries)
        print('trainloader name: ', names)
        print('trainloader len: ', lens)
        print('trainloader country: ', countries)
        break
