#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import rnn_wrapper, lens2mask

class LanguageModel(nn.Module):
    """ Container module with an encoder, a recurrent module, and a decoder.
    """
    def __init__(self, vocab_size=950, emb_size=400, hidden_size=400,
            num_layers=1, cell='lstm', pad_token_idx=0, dropout=0.5,
            decoder_tied=False, init_weight=0.2, **kargs):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell.upper() # RNN/LSTM/GRU
        self.pad_token_idx = pad_token_idx
        self.word_embed = nn.Embedding(vocab_size, emb_size)
        self.encoder = getattr(nn, self.cell)(
            self.emb_size, self.hidden_size, self.num_layers, bidirectional=False,
            batch_first=True, dropout=(dropout if self.num_layers > 1 else 0)
        )
        self.dropout_layer = nn.Dropout(dropout)
        decoder = nn.Linear(self.emb_size, self.vocab_size)
        if self.hidden_size != self.emb_size:
            self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.emb_size), decoder)
        else:
            self.decoder = decoder
        if decoder_tied:
            if self.hidden_size != self.emb_size:
                decoder.weight = self.word_embed.weight # shape: vocab_size, emb_size
            else:
                self.decoder.weight = self.word_embed.weight

        if init_weight:
            for p in self.parameters():
                p.data.uniform_(- init_weight, init_weight)
            self.word_embed.weight.data[self.pad_token_idx].zero_()

    def pad_embedding_grad_zero(self):
        self.word_embed.weight.grad[self.pad_token_idx].zero_()

    def forward(self, inputs, lens):
        inputs, lens = inputs[:, :-1], lens - 1
        emb = self.dropout_layer(self.word_embed(inputs)) # bsize, seq_length, emb_size
        outputs, _ = rnn_wrapper(self.encoder, emb, lens, self.cell)
        decoded = self.decoder(self.dropout_layer(outputs))
        scores = F.log_softmax(decoded, dim=-1)
        return scores

    def sent_logprob(self, inputs, lens, length_norm=False):
        ''' Given sentences, calculate the log-probability for each sentence
        @args:
            inputs(torch.LongTensor): sequence must contain <s> and </s> symbol
            lens(torch.LongTensor): length tensor
        @return:
            sent_logprob(torch.FloatTensor): logprob for each sent in the batch
        '''
        lens = lens - 1
        inputs, outputs = inputs[:, :-1], inputs[:, 1:]
        emb = self.dropout_layer(self.word_embed(inputs)) # bsize, seq_len, emb_size
        output, _ = rnn_wrapper(self.encoder, emb, lens, self.cell)
        decoded = self.decoder(self.dropout_layer(output))
        scores = F.log_softmax(decoded, dim=-1)
        logprob = torch.gather(scores, 2, outputs.unsqueeze(-1)).contiguous().view(output.size(0), output.size(1))
        sent_logprob = torch.sum(logprob * lens2mask(lens).float(), dim=-1)
        if length_norm:
            return sent_logprob / lens.float()
        else:
            return sent_logprob
