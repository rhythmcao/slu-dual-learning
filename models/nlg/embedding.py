#coding=utf8
import torch
import torch.nn as nn

class IntentEmbeddings(nn.Module):

    def __init__(self, emb_size, vocab, dropout=0.5):
        super(IntentEmbeddings, self).__init__()
        self.embed = nn.Embedding(vocab, emb_size)
        self.vocab = vocab
        self.emb_size = emb_size
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout_layer(self.embed(x))

class WordEmbeddings(nn.Module):

    def __init__(self, emb_size, vocab, pad_token_idx=0, dropout=0.5):
        super(WordEmbeddings, self).__init__()
        self.embed = nn.Embedding(vocab, emb_size)
        self.vocab = vocab
        self.emb_size = emb_size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pad_token_idx = pad_token_idx

    def forward(self, x):
        if x is None:
            return None
        return self.dropout_layer(self.embed(x))

    def pad_embedding_grad_zero(self):
        self.embed.weight.grad[self.pad_token_idx].zero_()
