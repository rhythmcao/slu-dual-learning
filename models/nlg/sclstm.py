#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCLSTM(nn.Module):
    """
        Semantically/Slots controlled LSTM, used in NLG decoder. See paper
            ``Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems, EMNLP 2015``
        Slots information are recorded in sclstm cell as slot state, mathematically,
            input gate: i_t = sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
            forget gate: f_t = sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
            output gate: o_t = sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
            reading gate: r_t = sigma(W_{ir} x_t + alpha * \sum_l W^l_{hr} h^l_{t-1})
            info: g_t = tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
            slot state: s_t = r_t * s_{t-1}
            cell state: c_t = f_t * c_{t-1} + i_t * g_t + tanh(W_{sc} s_t)
            hidden state: h_t = o_t * tanh(c_t)
        Args are almost the same with LSTM, except batch_fist=True, bidirectional=False are fixed.
        skip_connections(bool): if True, raw input is used in each SCLSTM layer,
            outputs of each layer are concatenated as final outputs.
    """
    def __init__(self, input_size, hidden_size, slot_size, num_layers, bias=True, dropout=0, skip_connections=False):
        super(SCLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.num_layers = num_layers if num_layers > 1 else 1
        self.bias = bias
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.skip_connections = skip_connections
        self.slot_cell = SlotCell(self.input_size, self.num_layers * self.hidden_size, self.slot_size, bias=self.bias)
        cell_list = [SCLSTMCell(self.input_size, self.hidden_size, self.slot_size, bias=self.bias)]
        middle_insize = self.hidden_size + self.input_size if self.skip_connections else self.hidden_size
        for i in range(num_layers - 1):
            cell_list.append(SCLSTMCell(middle_insize, self.hidden_size, self.slot_size, bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, inputs, h0_c0=None, s0=None, all_slot_hist=False, alpha=0.5):
        """
            @args:
                inputs(torch.FloatTensor): batch_size x seq_len x input_size
                h0_c0(tuple): hidden states and cell states, both torch.FloatTensor bsize x num_layers x hidden_size
                s0(torch.FloatTensor): slot states, batch_size x slot_size
                all_slot_hist(bool): whether return all slot states or final slot state
                alpha(float): coefficient combines word level pattern and phrase level pattern
            @return:
                outputs(torch.FloatTensor): batch_size x seq_len x (num_layers * hidden_size) if skip_connections else merely hidden_size
                ht_ct(tuple): final hidden states and cell states, both torch.FloatTensor bsize x num_layers x hidden_size
                st(torch.FloatTensor): batch_size [x seq_len ]x slot_size
        """
        bsize = inputs.size(0)
        if s0 is None:
            s0 = torch.zeros(bsize, self.slot_size).type_as(inputs).to(inputs.device)
        if h0_c0 is None:
            h0_c0 = (torch.zeros(bsize, self.num_layers, self.hidden_size).type_as(inputs).to(inputs.device), \
                torch.zeros(bsize, self.num_layers, self.hidden_size).type_as(inputs).to(inputs.device))
        st, ht, ct = s0, h0_c0[0], h0_c0[1]
        slot_hist, output_hist = [s0.unsqueeze(1)], []
        for t in range(inputs.size(1)):
            st = self.slot_cell(inputs[:, t, :], ht.contiguous().view(bsize, self.num_layers * self.hidden_size), st, alpha)
            if all_slot_hist:
                slot_hist.append(st.unsqueeze(dim=1))
            cur_input, next_ht, next_ct = inputs[:, t, :], [], []
            for l in range(self.num_layers):
                l_ht, l_ct = self.cell_list[l](cur_input, (ht[:, l, :], ct[:, l, :]), st)
                next_ht.append(l_ht.unsqueeze(dim=1))
                next_ct.append(l_ct.unsqueeze(dim=1))
                l_ht = self.dropout_layer(l_ht)
                cur_input = torch.cat([l_ht, inputs[:, t, :]], dim=1) if self.skip_connections else l_ht
            ht, ct = torch.cat(next_ht, dim=1), torch.cat(next_ct, dim=1)
            output = ht.contiguous().view(bsize, self.num_layers * self.hidden_size) if self.skip_connections else ht[:, -1, :]
            output_hist.append(output.unsqueeze(dim=1))
        outputs = torch.cat(output_hist, dim=1)
        if all_slot_hist:
            st = torch.cat(slot_hist, dim=1)
        return outputs, (ht, ct), st

class SCLSTMCell(nn.Module):
    """
        Semantically controlled LSTM cell, mathematically,
            input gate: i_t = sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
            forget gate: f_t = sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
            output gate: o_t = sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
            info: g_t = tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
            cell state: c_t = f_t * c_{t-1} + i_t * g_t + tanh(W_{sc} s_t)
            hidden state: h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_size, hidden_size, slot_size, bias=True):
        super(SCLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.bias = bias
        self.gates = nn.Linear(self.input_size + self.hidden_size, 4 * self.hidden_size, bias=self.bias)
        self.slot_linear = nn.Linear(self.slot_size, self.hidden_size, bias=self.bias)

    def forward(self, inputs, h_c, slot_state):
        """
            @args:
                inputs: bsize x input_size
                h_c: hidden state and cell state, (bsize x hidden_size, bsize x hidden_size)
                slot_state: bsize x slot_size
            @return:
                (next_h, next_c): hidden state and cell state of next timestep
        """
        hidden_state, cell_state = h_c
        combined = torch.cat([inputs, hidden_state], dim=1)
        all_gates = self.gates(combined)
        i_gate, f_gate, o_gate, info = torch.split(all_gates, self.hidden_size, dim=1)
        i_gate, f_gate, o_gate = torch.sigmoid(i_gate), torch.sigmoid(f_gate), torch.sigmoid(o_gate)
        info = torch.tanh(info)
        slot_info = torch.tanh(self.slot_linear(slot_state))
        next_c = f_gate * cell_state + i_gate * info + slot_info
        next_h = o_gate * torch.tanh(next_c)
        return next_h, next_c

class SlotCell(nn.Module):
    """
        Module calculates the update of slot cells. Mathematically,
            reading gate: r_t = sigma(W_{ir} x_t + alpha * \sum_l W^l_{hr} h^l_{t-1})
            slot state: s_t = r_t * s_{t-1}
    """
    def __init__(self, input_size, hidden_size, slot_size, bias=True):
        super(SlotCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.bias = True
        self.W_ir = nn.Linear(self.input_size, self.slot_size, bias=self.bias)
        self.W_hr = nn.Linear(self.hidden_size, self.slot_size, bias=self.bias)

    def forward(self, inputs, hidden_states, s_prev, alpha=0.5):
        """
            @args:
                inputs: input words x_t, bsize x input_size
                hidden_states: hidden states h_{t-1}, bsize x hidden_size, if num_layers > 1, hidden_size = num_layers * hidden_size
                s_prev: previous slot states, bsize x slot_size
                alpha: hyperparams that balance word pattern x_t and phrase pattern h_{t-1}
            @return:
                s_after: updated slot states, bsize x slot_size
        """
        r_t = torch.sigmoid(self.W_ir(inputs) + alpha * self.W_hr(hidden_states))
        s_after = s_prev * r_t
        return s_after
