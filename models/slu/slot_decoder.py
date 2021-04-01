#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotDecoder(nn.Module):
    """ Define standard linear + softmax decoder step.
    """
    def __init__(self, hidden_size, vocab_size, dropout=0.5, log_prob=True):
        super(SlotDecoder, self).__init__()
        self.decoder = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.log_prob = log_prob

    def forward(self, x):
        if self.log_prob:
            return F.log_softmax(self.decoder(self.dropout_layer(x)), dim=-1)
        else:
            return self.decoder(self.dropout_layer(x))

class SlotDecoderFocus(nn.Module):
    """
        Encoder-Decoder framework for slot decoder.
    """
    def __init__(self, emb_size, hidden_size, vocab_size, num_layers=1, cell='lstm', dropout=0.5):
        super(SlotDecoderFocus, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.cell = cell.upper()
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(p=dropout)
        self.dropout = dropout if self.num_layers > 1 else 0
        self.rnn_decoder = getattr(nn, self.cell)(
            self.emb_size + self.hidden_size * 2, self.hidden_size, num_layers=self.num_layers,
            bidirectional=False, batch_first=True, dropout=self.dropout
        )
        self.generator = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, tag_embeddings, memory, hidden_states):
        """
            @args:
                1. tag_embeddings: bsize x tgt_len x embed_size
                2. memory: bsize x src_len x hidden_size*2, src_len == tgt_len
                3. hidden_states: h_T, c_T, num_layers x bsize x hidden_size
            @return:
                1. log_prob: bsize x tgt_len x vocab_size
                2. hidden_states: h_T, c_T, num_layers x bsize x hidden_size
        """
        inputs = torch.cat([tag_embeddings, self.dropout_layer(memory)], dim=-1)
        outputs, hidden_states = self.rnn_decoder(inputs, hidden_states)
        log_prob = F.log_softmax(self.generator(self.dropout_layer(outputs)), dim=-1)
        return log_prob, hidden_states
