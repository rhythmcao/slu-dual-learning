#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import lens2mask

class StateTransition(nn.Module):

    def  __init__(self, slot_dim, intent_dim, hidden_size, num_layers=1, cell='LSTM', dropout=0.5):
        super(StateTransition, self).__init__()
        self.slot_dim = slot_dim
        self.intent_dim = intent_dim
        self.hidden_size = hidden_size
        self.attn = nn.Linear(slot_dim, intent_dim)
        self.affine = nn.Linear(slot_dim, hidden_size)
        self.num_layers = num_layers
        self.cell = cell.upper()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, slot_seq, slot_lens, intent_emb):
        """
        @args:
            slot_seq: bsize x max_slot_num x slot_dim
            slot_lens: tensor of # slot=value pair, bsize
            intent_emb: bsize x intent_dim
        @return:
            num_layers x bsize x hidden_size
        """
        if torch.sum(slot_lens).item() == 0:
            h = slot_lens.new_zeros(self.num_layers, slot_lens.size(0), self.hidden_size).float().contiguous()
            if self.cell == 'LSTM':
                c = h.new_zeros(h.size()).contiguous()
                return (h, c)
            return h
        slot_seq = self.dropout_layer(slot_seq)
        weights = torch.bmm(self.attn(slot_seq), intent_emb.unsqueeze(dim=-1)).squeeze(dim=-1)
        weights.masked_fill_(lens2mask(slot_lens) == 0, -1e8)
        a = F.softmax(weights, dim=-1)
        conxt = torch.bmm(a.unsqueeze(dim=1), slot_seq).squeeze(dim=1)
        h = torch.tanh(self.affine(conxt)).unsqueeze(dim=0).repeat(self.num_layers, 1, 1).contiguous()
        if self.cell == 'LSTM':
            c = h.new_zeros(h.size()).contiguous()
            return (h, c)
        return h