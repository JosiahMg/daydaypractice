import torch
from matplotlib import pyplot as plt


# 准备数据 y = 5x + 0.8
x = torch.rand([500, 1])
y = 5*x + 0.8
print(y.shape)


class LinearByHand:
    def __init__(self):
        # 初始化参数
        self.w = torch.rand([1, 1], requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    # 计算反向传播
    def loss_fn(self, y, y_predict):
        loss = (y - y_predict).pow(2).mean()
        for i in [self.w, self.b]:
            if i.grad is not None:
                i.grad.data.zero_()
            else:
                print('grad is none')
        loss.backward()
        return loss.data

    # 优化
    def optimizer(self, learning_rate):
        self.w.data -= learning_rate * self.w.grad.data
        self.b.data -= learning_rate * self.b.grad.data


model = LinearByHand()

for i in range(1000):
    y_predict = model.forward(x)
    loss = model.loss_fn(y, y_predict)

    if i % 100 == 0:
        print(i, loss)

    model.optimizer(0.02)

predict = x*model.w + model.b

plt.scatter(x.data.numpy(), y.data.numpy(), c='r')
plt.plot(x.data.numpy(), predict.data.numpy())
plt.show()

print('w', model.w)
print('b', model.b)
