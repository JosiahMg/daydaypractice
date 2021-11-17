"""
使用pytorch处理 wine:

"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math
import config


class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = pd.read_csv(config.wine_path, sep=',', header=[0])
        self.x = xy.iloc[:, 1:].to_numpy()
        self.y = xy.iloc[:, [0]].to_numpy()  # 注意0需要方括号，否则shape会不正确
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item], self.y[item]
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


# 定义转换tensor的类
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


dataset = WineDataset(transform=ToTensor())

BATCH_SIZE = 4
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 构造迭代器
datatiter = iter(dataloader)
data = next(datatiter)
features, labels = data
print(features, labels)

# epoch
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/BATCH_SIZE)
print(total_samples, n_iterations)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if i % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape} labels {labels.shape}')

