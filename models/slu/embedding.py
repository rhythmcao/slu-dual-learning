#coding=utf8
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from models.model_utils import lens2mask, PoolingFunction

class SLUEmbeddings(nn.Module):
    def __init__(self, emb_size, vocab, pad_token_idx=0, dropout=0.5):
        super(SLUEmbeddings, self).__init__()
        self.embed = nn.Embedding(vocab, emb_size)
        self.vocab = vocab
        self.emb_size = emb_size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pad_token_idx = pad_token_idx

    def forward(self, x):
        return self.dropout_layer(self.embed(x))

    def pad_embedding_grad_zero(self):
        self.embed.weight.grad[self.pad_token_idx].zero_()

class SLUPretrainedEmbedding(nn.Module):
    def __init__(self, dropout=0., subword_aggregation='attentive-pooling', lazy_load=False):
        super(SLUPretrainedEmbedding, self).__init__()
        self.plm = AutoModel.from_config(AutoConfig.from_pretrained('./data/.cache/bert-base-uncased')) \
            if lazy_load else AutoModel.from_pretrained('./data/.cache/bert-base-uncased')
        self.emb_size = self.plm.config.hidden_size
        self.subword_aggregation = SubwordAggregation(self.emb_size, subword_aggregation=subword_aggregation)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, plm_inputs):
        # inputs, lens = plm_inputs['inputs'], plm_inputs['lens']
        inputs = plm_inputs['inputs']
        lens = plm_inputs['lens']
        outputs = self.plm(**inputs)[0]
        subword_select_mask, subword_lens = plm_inputs['subword_select_mask'], plm_inputs['subword_lens']
        outputs = self.subword_aggregation(outputs, subword_select_mask, subword_lens, lens)
        return self.dropout_layer(outputs)

    def pad_embedding_grad_zero(self):
        pass

class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """
    def __init__(self, hidden_size, subword_aggregation='attentive-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, subword_select_mask, subword_lens, lens):
        selected_inputs = inputs.masked_select(subword_select_mask.unsqueeze(-1))
        reshaped_inputs = selected_inputs.new_zeros(subword_lens.size(0), max(subword_lens.tolist()), self.hidden_size)
        subword_mask = lens2mask(subword_lens)
        reshaped_inputs = reshaped_inputs.masked_scatter_(subword_mask.unsqueeze(-1), selected_inputs)
        # aggregate subword into word feats
        reshaped_inputs = self.aggregation(reshaped_inputs, mask=subword_mask)
        outputs = inputs.new_zeros(lens.size(0), max(lens.tolist()), self.hidden_size)
        outputs.masked_scatter_(lens2mask(lens).unsqueeze(-1), reshaped_inputs)
        return outputs
