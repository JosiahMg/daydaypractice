import torch


data = torch.randint(low=0, high=100, size=(64, 32, 100))

print(data[-1].shape)
