"""
实现attention

score(ht, hs) = :
1. dot(ht, hs)
2. general(ht, Wa, hs)
3. concat(va, tanh(Wa, [ht; hs]))
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from vocabs.vocab_number import vocab
import config


encoder_hidden_size = config.xhj_hidden_size
decoder_hidden_size = config.xhj_hidden_size


class Attention(nn.Module):
    def __init__(self, method='general'):
        super(Attention, self).__init__()
        assert method in ['dot', 'general', 'concat'], 'method error'
        self.method = method

        if self.method == 'general':
            self.Wa = nn.Linear(encoder_hidden_size,
                                decoder_hidden_size, bias=False)
        elif self.method == 'concat':
            self.Wa = nn.Linear(
                encoder_hidden_size+decoder_hidden_size, decoder_hidden_size, bias=False)
            self.Va = nn.Linear(decoder_hidden_size, 1)

    def forward(self, hidden_state, encoder_outputs):
        """

        :param hidden_state: (num_layer, batch_size, decoder_hidden_size)
        :param encoder_outputs: (batch, seq_len(1), encoder_hidden_size)
        :return:
        """
        if self.method == 'dot':
            # (batch_size, hidden_size, 1)
            hidden_state = hidden_state[[-1], :, :].permute(1, 2, 0)
            attention_weight = encoder_outputs.bmm(
                hidden_state).squeeze(-1)  # (batch_size, seq_len)
            attention_weight = F.softmax(attention_weight, dim=-1)

        elif self.method == 'general':
            # (batch, seq_len, decoder_hidden_size)
            encoder_outputs = self.Wa(encoder_outputs)
            # (batch, decoder_hidden_size, 1)
            hidden_state = hidden_state[[-1], :, :].permute(1, 2, 0)
            attention_weight = encoder_outputs.bmm(
                hidden_state).squeeze(-1)  # (batch, seq_len)
            attention_weight = F.softmax(
                attention_weight, dim=-1)  # (batch, seq_len)

        elif self.method == 'concat':
            # (batch, decoder_hidden_size)
            hidden_state = hidden_state[-1, :, :].squeeze(0)
            # (batch, seq_len, decoder_hidden_size)
            hidden_state = hidden_state.repeat(1, encoder_outputs.size(1), 1)
            # (batch, seq_len, hidden_size(decoder+encode)
            concated = torch.cat([hidden_state, encoder_outputs], dim=-1)
            attention_weight = self.Va(
                F.tanh(self.Wa(concated))).squeeze(-1)  # (batch, seq_len)
            attention_weight = F.softmax(attention_weight, dim=-1)

        return attention_weight
