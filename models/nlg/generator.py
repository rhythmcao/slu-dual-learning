#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLGGenerator(nn.Module):

    def __init__(self, input_size, gen_size, vocab, dropout=0.5):
        super(NLGGenerator, self).__init__()
        self.input_size = input_size
        self.gen_size = gen_size
        self.affine = nn.Linear(input_size, gen_size)
        self.gen_linear = nn.Linear(gen_size, vocab)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, feats):
        """
        @args:
            feats: bsize x tgt_len x gen_size
        @return:
            log_prob: bsize x tgt_len x vocab_size
        """
        log_prob = F.log_softmax(self.gen_linear(self.dropout_layer(self.affine(feats))), dim=-1)
        return log_prob

class NLGGeneratorCopy(nn.Module):

    def __init__(self, input_size, gen_size, vocab, slot_num=83, dropout=0.5):
        super(NLGGeneratorCopy, self).__init__()
        self.input_size = input_size
        self.gen_size = gen_size
        self.affine = nn.Linear(input_size, gen_size)
        self.gen_linear = nn.Linear(gen_size, vocab - slot_num)
        self.slot_num = slot_num
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, feats, pointer, gate_scores, copy_tokens):
        """
        @args:
            feats: bsize x tgt_len x gen_size
            pointer: bsize x tgt_len x max_slot_num
            gate_scores: copy or generate score, bsize x tgt_len x 1
            copy_tokens: used to map memory slots into corresponding target token idxs
                bsize x max_slot_num x vocab_size
        @return:
            log_prob: bsize x tgt_len x vocab_size
        """
        gen_prob = F.softmax(self.gen_linear(self.dropout_layer(self.affine(feats))), dim=-1)
        extra_zeros = torch.zeros(gen_prob.size(0), gen_prob.size(1), self.slot_num).type_as(gen_prob).to(gen_prob.device)
        gen_prob = torch.cat([extra_zeros, gen_prob], dim=-1)
        copy_prob = torch.bmm(pointer, copy_tokens) # bsize x tgt_len x vocab_size
        final_distribution = gate_scores * copy_prob + (1 - gate_scores) * gen_prob
        log_prob = torch.log(final_distribution + 1e-32)
        return log_prob
