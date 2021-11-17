from models.base_lib.encoder import Encoder
from models.base_lib.decoder import Decoder
import torch
import torch.nn as nn


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
            indices = self.decoder.evaluate(encoder_outputs, encoder_hidden)
            return indices
