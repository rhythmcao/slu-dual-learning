#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import lens2mask

class IntentDecoder(nn.Module):

    def __init__(self, hidden_size, vocab_size, cell='lstm', dropout=0.5, method='hiddenAttn'):
        super(IntentDecoder, self).__init__()
        self.h_c = cell.upper() == 'LSTM'
        self.hidden_size = hidden_size
        assert method in ['head+tail', 'hiddenAttn']
        self.method = method
        if self.method == 'hiddenAttn':
            self.Wa = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.Ua = nn.Conv1d(2 * self.hidden_size, self.hidden_size, 1, bias=False)
            self.Va = nn.Conv1d(self.hidden_size, 1, 1, bias=False)
        self.decoder = nn.Linear(self.hidden_size * 2, vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, *inputs):
        '''
        @args:
            hidden_states: RNN hidden states
            encoder_output: bsize x seqlen x hsize*2
            lens: length for each instance
        '''
        if self.method == 'head+tail':
            ht = inputs[0][0] if self.h_c else inputs[0]
            last = ht.size(0)
            index = [last - 2, last - 1]
            index = torch.tensor(index, dtype=torch.long, device=ht.device)
            ht = torch.index_select(ht, 0, index)
            sent = ht.transpose(0,1).contiguous().view(-1, 2 * self.hidden_size)
        elif self.method == 'hiddenAttn':
            hc, lstm_out, lens = inputs[0], inputs[1], inputs[2]
            ht = hc[0][-1] if self.h_c else hc[-1]
            hiddens = lstm_out.transpose(1, 2)
            c1 = self.Wa(self.dropout_layer(ht))
            c2 = self.Ua(self.dropout_layer(hiddens))
            c3 = c1.unsqueeze(2).repeat(1, 1, lstm_out.size(1))
            c4 = torch.tanh(c3 + c2)
            e = self.Va(c4).squeeze(1)
            e.masked_fill_(lens2mask(lens) == 0, -float('inf'))
            a = F.softmax(e, dim=1)
            sent = torch.bmm(hiddens, a.unsqueeze(2)).squeeze(2)
        else:
            raise ValueError('Unrecognized intent detection method!')
        return F.log_softmax(self.decoder(self.dropout_layer(sent)), dim=1)