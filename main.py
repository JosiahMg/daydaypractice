import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset


class NumVocab:
    def transform(self, max_len):
        res = []


class NumDataset(Dataset):
    def __init__(self, vocab=None):
        self.data = np.random.randint(0, 1e8, size=[500000])

    def __getitem__(self, index: int):
        res = list(map(int, list(str(self.data[index]))))
        return res

    def __len__(self) -> int:
        return self.data.shape[0]


num_dataset = NumDataset()


print(num_dataset[0])
print(len(num_dataset))
