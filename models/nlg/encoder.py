#coding=utf8
import torch
import torch.nn as nn
from models.model_utils import lens2mask, rnn_wrapper, PoolingFunction

class SlotEncoder(nn.Module):

    def __init__(self, emb_size, hidden_size, num_layers, cell='lstm', dropout=0.5, slot_aggregation='attentive-pooling'):
        super(SlotEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell.upper()
        self.dropout = dropout if self.num_layers > 1 else 0
        self.slot_encoder = getattr(nn, self.cell)(
            self.emb_size, self.hidden_size, num_layers=self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout
        )
        self.slot_aggregation = PoolingFunction(self.hidden_size * 2, self.hidden_size * 2, method=slot_aggregation)

    def forward(self, slot_emb, slot_lens, lens):
        """
        @args:
            slot_emb: [total_slot_num, max_slot_word_len, emb_size]
            slot_lens: slot_num for each training sample, [bsize]
            lens: seq_len for each ${slot}=value sequence, [total_slot_num]
        @return:
            slot_feats: bsize, max_slot_num, hidden_size * 2
        """
        if slot_emb is None or torch.sum(slot_lens).item() == 0:
            # set seq_len dim to 1 due to decoder attention computation
            return torch.zeros(slot_lens.size(0), 1, self.hidden_size * 2, dtype=torch.float).to(slot_lens.device)
        slot_outputs, _ = rnn_wrapper(self.slot_encoder, slot_emb, lens, self.cell)
        slot_outputs = self.slot_aggregation(slot_outputs, lens2mask(lens))
        chunks = slot_outputs.split(slot_lens.tolist(), dim=0) # list of [slot_num x hidden_size]
        max_slot_num = torch.max(slot_lens).item()
        padded_chunks = [torch.cat([each, each.new_zeros(max_slot_num - each.size(0), each.size(1))], dim=0) for each in chunks]
        # bsize x max_slot_num x hidden_size
        slot_feats = torch.stack(padded_chunks, dim=0)
        return slot_feats

class NLGEncoder(nn.Module):

    def __init__(self, slot_encoder, input_size, hidden_size, num_layers=1, cell='lstm', dropout=0.5):
        super(NLGEncoder, self).__init__()
        self.slot_encoder = slot_encoder
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell.upper()
        self.dropout = dropout if self.num_layers > 1 else 0
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_encoder = getattr(nn, self.cell)(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout
        )

    def forward(self, slot_emb, slot_lens, lens):
        """
        @args:
            slot_emb: [total_slot_num, max_slot_word_len, emb_size]
            slot_lens: slot_num for each training sample, [bsize]
            lens: seq_len for each ${slot}=value sequence, [total_slot_num]
        @return:
            slot_feats: bsize, max_slot_num, hidden_size * 2
        """
        if slot_emb is None or torch.sum(slot_lens).item() == 0:
            # set seq_len dim to 1 due to decoder attention computation
            return torch.zeros(slot_lens.size(0), 1, self.hidden_size * 2, dtype=torch.float).to(slot_lens.device)
        else:
            slot_feats = self.slot_encoder(slot_emb, slot_lens, lens)
            slot_outputs, _ = rnn_wrapper(self.rnn_encoder, self.dropout_layer(slot_feats), slot_lens, self.cell)
            return slot_outputs
