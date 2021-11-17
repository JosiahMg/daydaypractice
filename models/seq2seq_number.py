"""
将字符串"12345"->"123450"
每个字符串后面加一个'0'
步骤:
1. 构造词典
2. 构造dataloader
3. 构造模型
4. 训练

"""
from datasets.dataset_number import train_data_loader, vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from models.base_lib.seq2seq import Seq2seq
import config


seq2seq = Seq2seq().to(config.device)
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=0.001)


def train_model(epoch):
    """
    input.shape: (batch, seq_len)
    target.shape: (batch, seq_len)
    input_len: (batch,)
    target_len: (batch,)
    """
    bar = tqdm(enumerate(train_data_loader), total=len(
        train_data_loader), ascii=True, desc='train')

    for index, (inputs, target, input_len, target_len) in bar:
        inputs, target, input_len, target_len = inputs.to(config.device), target.to(
            config.device), input_len.to(config.device), target_len.to(config.device)

        optimizer.zero_grad()
        # decoder_outputs.shape: (batch, seq_len, vocab_size)
        decoder_outputs, _ = seq2seq(inputs, target, input_len, target_len)

        # (batch*seq, vocab_size)
        decoder_outputs = decoder_outputs.view(-1, len(vocab))

        target = target.view(-1)  # (batch*seq,)

        loss = F.nll_loss(decoder_outputs, target, ignore_index=vocab.PAD)

        loss.backward()
        optimizer.step()

        if index % 500 == 0:
            bar.set_description('epoch: {} idx:{} loss:{:.4f}'.format(
                epoch, index, loss.item()))
            torch.save(seq2seq.state_dict(), config.number_model_path)
            torch.save(optimizer.state_dict(), config.number_optim_path)


def test_model():
    # 1. 准备数据
    data = [str(i) for i in np.random.randint(0, 1e8, size=[100])]
    data = sorted(data, key=lambda x: len(x), reverse=True)
    target = [i+'0' for i in data]
    input_length = torch.LongTensor([len(i) for i in data]).to(config.device)
    inputs = torch.LongTensor(
        [vocab.transform(list(i), config.number_seq_max_len) for i in data]).to(config.device)

    # 2. 加载模型
    seq2seq = Seq2seq()
    seq2seq = seq2seq.to(config.device)
    seq2seq.load_state_dict(torch.load(config.number_model_path))

    # 3. 预测
    indices = seq2seq.evaluate(inputs, input_length)
    indices = np.array(indices).transpose()

    # 4. 反序列化
    result = []
    for line in indices:
        temp_result = vocab.inverse_transform(line)
        cur_line = ''
        for word in temp_result:
            if word == vocab.EOS_TAG:
                break
            cur_line += word
        result.append(cur_line)
    print(data[:10])
    print(result[:10])
    acc = sum(i == j for i, j in zip(target, result))/len(target)
    print('Accuracy: ', acc)


def test_num_model():
    x_inputs = [str(i) for i in np.random.randint(0, 1e8, size=[100])]
    x_inputs = sorted(x_inputs, lambda x: len(x), reverse=True)


if __name__ == '__main__':
    for i in range(3):
        train_model(i)
    test_model()
