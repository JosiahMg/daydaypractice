import torch
import torch.nn as nn

# 定义数据
x = torch.rand([500, 1])
y = x*5 + 0.8


# 定义模型
class LinearByTorch(nn.Module):
    def __init__(self):
        super(LinearByTorch, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearByTorch()

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.10)


for epoch in range(1000):
    y_pre = model(x)
    loss = loss_fn(y, y_pre)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%100 == 0:
        print(f'epoch: {epoch} loss: {loss.item()} w: {model.linear.weight.item()} b: {model.linear.bias.item()}')

