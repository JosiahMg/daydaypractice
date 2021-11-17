import torch
import torch.nn as nn


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
