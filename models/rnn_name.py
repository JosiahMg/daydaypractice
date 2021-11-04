"""
使用rnn模型预测人名是哪个国家

输入: "Maclen"
输出: "Janpan"

知识点： 
1. rnn
2. pad_packed_sequence & pack_padded_sequence

步骤:
1. 构造词典
2. 构造dataset & dataloader
3. 构造模型
4. 训练
5. 测试
6. 绘制loss

"""
from numpy import core
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import config
import pandas as pd
from data_processing.dataset_name import NameDataset, collate_fn, make_tensors
from data_processing.vocab_name import NameVocab
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# 1. build dictionary
vocab = NameVocab()

BATCH_SIZE = 128
N_EPOCHS = 50
N_CHARS = 128
HIDDEN_SIZE = 256
N_COUNTRY = vocab.country_size
N_LAYER = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 2. create dataloader
trainset = NameDataset(vocab=vocab, is_train=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=collate_fn)

testset = NameDataset(vocab=vocab, is_train=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_fn)

# 3. build model


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size,
                                n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*self.n_directions, output_size)

    def _init_hidden(self, batchSize, requries_grad=True):
        weight = next(self.parameters())
        h0 = weight.new_zeros((self.n_layers*self.n_directions,
                               batchSize, self.hidden_size), requires_grad=requries_grad)
        c0 = weight.new_zeros((self.n_layers*self.n_directions,
                               batchSize, self.hidden_size), requires_grad=requries_grad)
        return h0, c0

    def forward(self, input, seq_lengths):
        """
        input.shape: (batch_size, seq_len)
        seq_lengths: (batch_size,)  must be sorted by descendent
        """
        # input shape: B*S->S*B
        input = input.t()  # (seq_len, batch_size)
        batch_size = input.size(1)

        h0 = self._init_hidden(batch_size)
        embedding = self.embedding(input)  # (seq_len, batch_size, emb_size)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        # output.shape: (seq_len, batch_size, hidden_size*n_directions)
        # hidden.shape: (n_layers*n_directions, batch_size, hidden_size)
        output, hidden = self.gru(gru_input, h0[0])
        if self.n_directions == 2:
            hidden_cat = torch.cat(
                [hidden[-1], hidden[-2]], dim=1)  # (batch, hidden_size*2)
        else:
            hidden_cat = hidden[-1]  # (batch, hidden_size)

        fc_output = self.fc(hidden_cat)
        return fc_output


# 4.train model
def trainModel():
    total_loss = 0.
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, targets = make_tensors(names, countries)
        inputs = inputs.to(device)
        seq_lengths = seq_lengths.to(device)
        targets = targets.to(device)

        output = classifier(inputs, seq_lengths)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'Epoch {epoch}', end='')
            print(f'[{i*len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss/(i*len(inputs))}')
    return total_loss


# 5.test model
def testModel():
    correct = 0.
    total = len(testset)
    print('Evaluating trained model ...')

    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            inputs = inputs.to(device)
            seq_lengths = seq_lengths.to(device)
            target = target.to(device)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100*correct/total)
        print(f'Test set: Accuarcy {correct}/{total} {percent}%')
    return correct/total


def plot_acc(acc_list):
    epoch = np.arange(1, len(acc_list)+1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    classifier = RNNClassifier(
        N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    print('Training for %d epochs...' % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    plot_acc(acc_list)
