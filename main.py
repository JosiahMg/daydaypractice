import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import config
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = pd.read_csv(config.wine_path, sep=',', header=[0])
        self.x = xy.iloc[:, 1:].to_numpy()
        self.y = xy.iloc[:, [0]].to_numpy()
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item], self.y[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.y)



class ToTensor:
    def __call__(self, sample):
        x, y = sample
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        return x, y

dataset = WineDataset()
total_samples = len(dataset)
BATCH_SIZE = 4
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


n_iterations = math.ceil(total_samples/BATCH_SIZE)

for epoch in range(2):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if i % 5 == 0:
            print(f'epoch {epoch+1}/{2}, step {i+1}/{n_iterations}, inputs {inputs.shape} labels {labels.shape}')
