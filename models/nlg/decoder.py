#coding=utf8
import torch
import torch.nn as nn
from models.nlg.sclstm import SCLSTM

class NLGDecoderSCLSTM(nn.Module):

    def __init__(self, emb_size, hidden_size, intent_size, slot_num, num_layers, attn=None, dropout=0.5, skip_connections=False):
        super(NLGDecoderSCLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.intent_size = intent_size
        self.slot_num = slot_num
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(p=dropout)
        self.dropout = dropout if self.num_layers > 1 else 0
        self.skip_connections = skip_connections
        self.rnn_decoder = SCLSTM(self.emb_size + self.intent_size, self.hidden_size, self.slot_num,
                num_layers=self.num_layers, dropout=self.dropout, skip_connections=self.skip_connections)
        self.attn = attn

    def forward(self, inputs, intent_emb, memory, decoder_states, slot_states, slot_lens, all_slot_hist=False):
        """
        @args:
            inputs: bsize x tgt_len x (emb_size + intent_size)
            intent_emb: bsize x intent_size
            memory: bsize x max_slot_num x slot_dim
            decoder_states: bsize x num_layers x hidden_size
            slot_states: bsize x slot_num
            slot_lens: slot number of each batch sample, bsize
            all_slot_hist(bool): whether return last slot state or all the slot history
        @return:
            feats: bsize x tgt_lens x (dec_dim + enc_dim)
            decoder_states: bsize x num_layers x hidden_size
            slot_states: bsize [x tgt_lens ]x slot_num
        """
        inputs = torch.cat([inputs, intent_emb.unsqueeze(1).repeat(1, inputs.size(1), 1)], dim=-1)
        output_hiddens, decoder_states, slot_states = self.rnn_decoder(inputs, decoder_states, slot_states, all_slot_hist)
        context = []
        for i in range(output_hiddens.size(1)):
            tmp_context, _ = self.attn(memory, output_hiddens[:, i, :], slot_lens)
            context.append(tmp_context)
        context = torch.cat(context, dim=1)
        feats = torch.cat([self.dropout_layer(output_hiddens), context], dim=-1)
        return feats, decoder_states, slot_states

class NLGDecoderSCLSTMCopy(nn.Module):

    def __init__(self, emb_size, hidden_size, intent_size, slot_num, num_layers, attn=None, dropout=0.5, skip_connections=False):
        super(NLGDecoderSCLSTMCopy, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.intent_size = intent_size
        self.slot_num = slot_num
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(p=dropout)
        self.dropout = dropout if self.num_layers > 1 else 0
        self.skip_connections = skip_connections
        self.rnn_decoder = SCLSTM(self.emb_size + self.intent_size, self.hidden_size, self.slot_num,
            num_layers=self.num_layers, dropout=self.dropout, skip_connections=self.skip_connections)
        self.attn = attn
        out_hidden_size = self.num_layers * self.hidden_size if self.skip_connections else self.hidden_size
        self.gate = nn.Linear(out_hidden_size + self.attn.enc_dim + self.emb_size + self.intent_size, 1)

    def forward(self, inputs, intent_emb, memory, decoder_states, slot_states, slot_lens, all_slot_hist=False):
        """
            @args:
                inputs: bsize x tgt_len x (emb_size + intent_size)
                intent_emb: bsize x intent_size
                memory: bsize x max_slot_num x slot_dim
                decoder_states: bsize x num_layers x hidden_size
                slot_states: bsize x slot_num
                slot_lens: slot number of each batch sample, bsize
                all_slot_hist(bool): whether return last slot state or all the slot history
            @return:
                feats: bsize x tgt_lens x (dec_dim + enc_dim)
                decoder_states: bsize x num_layers x hidden_size
                slot_states: bsize [x tgt_lens ]x slot_num
                pointer: bsize x tgt_lens x max_slot_num
                gate_scores: bsize x tgt_lens x 1
        """
        inputs = torch.cat([inputs, intent_emb.unsqueeze(1).repeat(1, inputs.size(1), 1)], dim=-1)
        output_hiddens, decoder_states, slot_states = self.rnn_decoder(inputs, decoder_states, slot_states, all_slot_hist)
        context, pointer = [], []
        for i in range(output_hiddens.size(1)):
            tmp_context, tmp_pointer = self.attn(memory, output_hiddens[:, i, :], slot_lens)
            context.append(tmp_context)
            pointer.append(tmp_pointer.unsqueeze(dim=1))
        context, pointer = torch.cat(context, dim=1), torch.cat(pointer, dim=1)
        feats = torch.cat([self.dropout_layer(output_hiddens), context], dim=-1)
        gate_scores = torch.sigmoid(self.gate(torch.cat([feats, inputs], dim=-1)))
        return feats, decoder_states, slot_states, pointer, gate_scores