import torch
import torch.nn as nn
from vocabs.vocab_number import vocab
import config
import numpy as np
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=config.number_emb_size, padding_idx=vocab.PAD)

        self.gru = torch.nn.GRU(input_size=config.number_emb_size, num_layers=config.number_num_layer,
                                hidden_size=config.number_hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(config.number_hidden_size, len(vocab))

    def forward(self, target, encoder_hidden):
        # 1. encoder的hidden输出最为decoder的输入
        decoder_hidden = encoder_hidden  # (n_layer*direction, batch, hidden)

        batch_size = target.size(0)
        decoder_input = torch.LongTensor(torch.ones(
            [batch_size, 1], dtype=torch.int64)*vocab.SOS).to(config.device)  # (batch, 1)

        decoder_outputs = torch.zeros(
            [batch_size, config.number_seq_max_len+1, len(vocab)]).to(config.device)

        for t in range(config.number_seq_max_len+1):
            # decoder_output.shape: (batch, hidden_size)
            # decoder_hidden.shape: (n_layer*direction, batch, hidden_size)
            decoder_output_t, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            if np.random.random() > config.number_teach_forcing_rate:
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

    def evaluate(self, encoder_outputs, encoder_hidden):
        decoder_hidden = encoder_hidden  # (n_layer*direction, batch, hidden)

        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones(
            [batch_size, 1], dtype=torch.int64)*vocab.SOS).to(config.device)  # (batch, 1)

        indices = []
        for _ in range(config.number_seq_max_len+5):
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
