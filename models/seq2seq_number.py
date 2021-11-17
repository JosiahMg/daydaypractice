"""
将字符串"12345"->"123450"
每个字符串后面加一个'0'
步骤:
1. 构造词典
2. 构造dataloader
3. 构造模型
4. 训练

"""
from datasets.dataset_number import seq_max_len, train_data_loader, vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

emb_size = 100
num_dict = len(vocab)
num_layer = 1
hidden_size = 64
teach_forcing_rate = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_dict, embedding_dim=emb_size, padding_idx=vocab.PAD)

        self.gru = torch.nn.GRU(input_size=emb_size, num_layers=num_layer,
                                hidden_size=hidden_size, batch_first=True, bidirectional=False)

    def forward(self, x, x_len):
        embeded = self.embedding(x)  # (batch, max_len, emb_size)
        embeded = torch.nn.utils.rnn.pack_padded_sequence(
            embeded, x_len, batch_first=True)
        # output.shape: (batch, seq, hidden_size)
        # hidden.shape: (seq, batch, hidden_size)
        output, hidden = self.gru(embeded)
        # output.shape: (batch, seq, hidden_size)
        # output_len.shape: (batch,)
        output, output_len = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, padding_value=vocab.PAD)

        return output, hidden


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_dict, embedding_dim=emb_size, padding_idx=vocab.PAD)

        self.gru = torch.nn.GRU(input_size=emb_size, num_layers=num_layer,
                                hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_dict)

    def forward(self, target, encoder_hidden):
        # 1. encoder的hidden输出最为decoder的输入
        decoder_hidden = encoder_hidden  # (n_layer*direction, batch, hidden)

        batch_size = target.size(0)
        decoder_input = torch.LongTensor(torch.ones(
            [batch_size, 1], dtype=torch.int64)*vocab.SOS).to(device)  # (batch, 1)

        decoder_outputs = torch.zeros(
            [batch_size, seq_max_len+1, num_dict]).to(device)

        for t in range(seq_max_len+1):
            # decoder_output.shape: (batch, hidden_size)
            # decoder_hidden.shape: (n_layer*direction, batch, hidden_size)
            decoder_output_t, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            if np.random.random() > teach_forcing_rate:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                # value.shape: (batch, 1)
                # index.shape: (batch, 1)
                value, index = torch.topk(decoder_output_t, k=1, dim=-1)
                decoder_input = index
        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        decoder_input_emb = self.embedding(
            decoder_input)  # (batch, 1, emb_size)
        out, decoder_hidden = self.gru(decoder_input_emb, decoder_hidden)
        out = out.squeeze(1)  # batch, hidden_size
        output = F.log_softmax(self.fc(out), dim=-1)  # batch, vocab_size
        return output, decoder_hidden

    def evaluate(self, encoder_hidden):
        decoder_hidden = encoder_hidden  # (n_layer*direction, batch, hidden)

        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones(
            [batch_size, 1], dtype=torch.int64)*vocab.SOS).to(device)  # (batch, 1)

        indices = []
        for _ in range(seq_max_len+5):
            # decoder_output_t.shape: (batch, vocab_size)
            # decoder_hidden.shape: (n_layer*bidirection, batch, hidden_size)
            decoder_output_t, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            # value.shape: (batch, 1)
            # index.shape: (batch, 1)
            value, index = torch.topk(decoder_output_t, k=1, dim=-1)
            decoder_input = index
            # if index.item() == vocab.EOS:
            #     break
            indices.append(index.squeeze(-1).cpu().detach().numpy())
        return indices


class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, target, input_len, target_len):
        encoder_outputs, encoder_hidden = self.encoder(input, input_len)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden)
        return decoder_outputs, decoder_hidden

    def evaluate(self, inputs, input_length):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(
                inputs, input_length)
            indices = self.decoder.evaluate(encoder_hidden)
            return indices


seq2seq = Seq2seq().to(device)
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
        inputs, target, input_len, target_len = inputs.to(device), target.to(
            device), input_len.to(device), target_len.to(device)
        optimizer.zero_grad()
        # decoder_outputs.shape: (batch, seq_len, vocab_size)
        decoder_outputs, _ = seq2seq(inputs, target, input_len, target_len)
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(
            0)*decoder_outputs.size(1), -1)  # (batch*seq, vocab_size)
        target = target.view(-1)  # (batch*seq,)
        loss = F.nll_loss(decoder_outputs, target, ignore_index=vocab.PAD)
        loss.backward()
        optimizer.step()
        if index % 500 == 0:
            bar.set_description('epoch: {} idx:{} loss:{:.4f}'.format(
                epoch, index, loss.item()))
            torch.save(seq2seq.state_dict(), 'seq2seq_number.model')
            torch.save(optimizer.state_dict(), 'seq2seq_number.opt')


def test_model():
    # 1. 准备数据
    data = [str(i) for i in np.random.randint(0, 1e8, size=[100])]
    data = sorted(data, key=lambda x: len(x), reverse=True)
    target = [i+'0' for i in data]
    input_length = torch.LongTensor([len(i) for i in data]).to(device)
    inputs = torch.LongTensor(
        [vocab.transform(list(i), seq_max_len) for i in data]).to(device)

    # 2. 加载模型
    seq2seq = Seq2seq()
    seq2seq = seq2seq.to(device)
    seq2seq.load_state_dict(torch.load('seq2seq_number.model'))

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
