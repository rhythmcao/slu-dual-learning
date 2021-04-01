#coding=utf8
import torch
import torch.nn as nn

class StateTransition(nn.Module):

    def __init__(self, num_layers, cell='lstm'):
        """ Transform encoder final hidden states to decoder initial hidden states
        """
        super(StateTransition, self).__init__()
        self.num_layers = num_layers
        self.cell = cell.upper()

    def forward(self, hidden_states):
        index_slices = [2 * i + 1 for i in range(self.num_layers)]  # from reversed path
        index_slices = torch.tensor(index_slices, dtype=torch.long, device=hidden_states[0].device)
        if self.cell == 'LSTM':
            enc_h, enc_c = hidden_states
            dec_h = torch.index_select(enc_h, 0, index_slices)
            dec_c = torch.index_select(enc_c, 0, index_slices)
            hidden_states = (dec_h.contiguous(), dec_c.contiguous())
        else:
            enc_h = hidden_states
            dec_h = torch.index_select(enc_h, 0, index_slices)
            hidden_states = dec_h.contiguous()
        return hidden_states