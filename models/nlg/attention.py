#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import lens2mask

class Attention(nn.Module):
    METHODS = ['dot', 'feedforward']

    def __init__(self, enc_dim, dec_dim, dropout=0.5, method='feedforward'):

        super(Attention, self).__init__()
        self.enc_dim, self.dec_dim = enc_dim, dec_dim
        assert method in Attention.METHODS
        self.method = method
        if self.method == 'dot':
            self.Wa = nn.Linear(self.enc_dim, self.dec_dim, bias=False)
        else:
            self.Wa = nn.Linear(self.enc_dim + self.dec_dim, self.dec_dim, bias=False)
            self.Va = nn.Linear(self.dec_dim, 1, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, hiddens, decoder_state, slot_lens):
        '''
        @args:
            hiddens : bsize x max_slot_num x enc_dim
            decoder_state : bsize x dec_dim
            slot_lens : slot number for each batch, bsize
        @return:
            context : bsize x 1 x enc_dim
            a : normalized coefficient, bsize x max_slot_num
        '''
        decoder_state = self.dropout_layer(decoder_state)
        if self.method == 'dot':
            m = self.Wa(self.dropout_layer(hiddens))
            m = m.transpose(-1, -2)
            e = torch.bmm(decoder_state.unsqueeze(1), m).squeeze(dim=1)
        else:
            d = decoder_state.unsqueeze(dim=1).repeat(1, hiddens.size(1), 1)
            e = self.Wa(torch.cat([d, self.dropout_layer(hiddens)], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
        masks = lens2mask(slot_lens)
        if masks.size(1) < e.size(1):
            masks = torch.cat([masks, torch.zeros(masks.size(0), e.size(1) - masks.size(1)).type_as(masks).to(masks.device)], dim=1)
        e.masked_fill_(masks == 0, -1e8)
        a = F.softmax(e, dim=1)
        context = torch.bmm(a.unsqueeze(1), hiddens)
        return context, a