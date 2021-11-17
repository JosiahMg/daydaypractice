"""
进行梯度裁剪，防止梯度爆炸
1. 在反向传播梯度计算结束后
2. 在梯度更新之前

import torch.nn as nn

clip = 0.01
loss.backward()  # 计算梯度
nn.utils.clip_grad_norm_(model.parameters, clip)  # 裁剪
optimizer.step()  # 梯度更新

"""

import torch.nn as nn

clip = 0.01
model = nn.RNN(input_size=100, hidden_size=100)

# loss.backward()
nn.utils.clip_grad_norm_(model.parameters, clip)
# optimizer.step()
