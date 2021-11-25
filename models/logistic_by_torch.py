""" 
使用skearn自带的load_breast_cancer数据集训练逻辑回归
"""


# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#  - forward pass： compute prediction and loss
#  - backward pass:gradients
#  - update weights


import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 0) prepare data

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

n_input_features = n_features
n_output_features = 1


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b, sigmoid at the end


class LogisticRegress(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super(LogisticRegress, self).__init__()
        self.linear = nn.Linear(n_input_features, n_output_features)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegress(n_input_features, n_output_features)

# 2) loss and optimizer
learning_rate = 0.05
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training data
num_epochs = 1000

for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # backward pass : gradients
    loss.backward()

    # update weights
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch = {epoch+1} loss = {loss.item():.4f}')

with torch.no_grad():
    model.eval()  # 作用是让dropout全部生效
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum().item() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
