""" 
使用skearn自带的load_breast_cancer数据集训练一个逻辑回归
"""


from sklearn import datasets
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 数据集

data = datasets.load_breast_cancer()

x = data['data'].astype(np.float32)
y = data['target'].astype(np.float32)

sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train).view(-1, 1)
y_test = torch.from_numpy(y_test).view(-1, 1)

print(x_train.size())
print(x_test.size())
print(y_train.size())
print(y_test.size())
# 定义模型


class LogisticModel(nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(x_train.size(1), y_train.size(1))

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)


# 3. 构造模型和损失函数
model = LogisticModel()
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. train


def train():
    y_p = model(x_train)
    loss = loss_fn(y_p, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate():
    with torch.no_grad():
        y_pre = model(x_test)  # (batch, 1)
        y_pre = y_pre.round()
        print(y_pre.eq(y_test).sum().item()/y_test.size(0))


if __name__ == '__main__':
    for epoch in range(1000):
        loss = train()
        if epoch % 200 == 0:
            print(f'epoch {epoch}, loss is {loss:.3f}')

    evaluate()
