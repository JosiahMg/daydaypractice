"""
torch的模型保存和加载
"""
import torch
import torch.nn as nn

"""
方法一
    torch.save(model, PATH)
    model = torch.load(PATH)
    model.eval()
"""

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)
# train your medel...

for param in model.parameters():
    print(param)

# save model
FILE = "model.pth"
torch.save(model, FILE)

# load model
model = torch.load(FILE)

# 防止模型参数发生变化
model.eval()
for param in model.parameters():
    print(param)

"""
    方法二
    torch.save(model.state_dict(), PATH)
    # model must be created again with parameters
    model = Model(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
"""
# save model
torch.save(model.state_dict(), FILE)

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))

# 防止模型参数发生变化
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)

"""
方法三：

定义一个字典，保存多个参数到模型
"""
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# print(optimizer.state_dict())


checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}
# 保存三种数据到模型
torch.save(checkpoint, "checkpoint.pth")

# 加载模型
loaded_checkpoint = torch.load("checkpoint.pth")
# 载入epcho数据
epoch = loaded_checkpoint['epoch']
print(epoch)

# 定义模型和优化器
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)


# 将保存的模型数据载入到模型和优化器中
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])
model.eval()
for param in model.parameters():
    print(param)
