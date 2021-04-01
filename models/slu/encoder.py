#coding=utf8
import torch
import torch.nn as nn
from models.model_utils import rnn_wrapper

class RNNEncoder(nn.Module):
    """ Core encoder is a stack of N RNN layers
    """
    def __init__(self, emb_size, hidden_size, num_layers, cell="lstm", dropout=0.5):
        super(RNNEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0
        self.cell = cell.upper()
        self.rnn_encoder = getattr(nn, self.cell)(self.emb_size, self.hidden_size, num_layers=self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout)

    def forward(self, x, src_lens):
        """
            Pass the input (and src_lens) through each RNN layer in turn.
        """
        out, hidden_states = rnn_wrapper(self.rnn_encoder, x, src_lens, self.cell)
        return out, hidden_states