#coding=utf8
from models.nlg.embedding import IntentEmbeddings, WordEmbeddings
from models.nlg.encoder import SlotEncoder, NLGEncoder
from models.nlg.init_decoder import StateTransition
from models.nlg.attention import Attention
from models.nlg.decoder import NLGDecoderSCLSTM, NLGDecoderSCLSTMCopy
from models.nlg.generator import NLGGeneratorCopy, NLGGenerator
from models.nlg.nlg_model_sclstm import SCLSTMModel
from models.nlg.nlg_model_sclstm_copy import SCLSTMCopyModel

def construct_models(**kargs):
    constructor = {
        'sclstm': construct_model_sclstm,
        'sclstm+copy': construct_model_sclstm_copy
    }
    model_type = constructor[kargs.pop('model_type', 'sclstm+copy')]
    return model_type(**kargs)

def construct_model_sclstm(emb_size=400, vocab_size=952, intent_num=18, slot_num=83, hidden_size=256, num_layers=1, pad_token_idxs={'word': 0},
        cell='lstm', dropout=0.5, slot_aggregation='attentive-pooling', init_weight=0.2, **kargs):
    intent_embed = IntentEmbeddings(emb_size, intent_num, dropout=dropout)
    word_embed = WordEmbeddings(emb_size, vocab_size, pad_token_idxs['word'], dropout=dropout)
    slot_encoder = SlotEncoder(emb_size, hidden_size, num_layers, cell=cell, dropout=dropout, slot_aggregation=slot_aggregation)
    encoder = NLGEncoder(slot_encoder, hidden_size * 2, hidden_size, num_layers, cell=cell, dropout=dropout)
    enc2dec = StateTransition(hidden_size * 2, emb_size, hidden_size, num_layers, cell=cell, dropout=dropout)
    attn = Attention(hidden_size * 2, hidden_size, dropout=dropout, method='feedforward')
    decoder = NLGDecoderSCLSTM(emb_size, hidden_size, emb_size, slot_num, num_layers, attn=attn, dropout=dropout)
    generator = NLGGenerator(hidden_size * 2 + hidden_size, hidden_size, vocab_size, dropout=dropout)
    m = SCLSTMModel(intent_embed, word_embed, encoder, decoder, enc2dec, generator)
    if init_weight:
        for p in m.parameters():
            p.data.uniform_(- init_weight, init_weight)
        m.word_embed.embed.weight.data[pad_token_idxs['word']].zero_()
    return m

def construct_model_sclstm_copy(emb_size=400, vocab_size=952, intent_num=18, slot_num=83, hidden_size=256, num_layers=1,
        pad_token_idxs={'word': 0}, cell='lstm', dropout=0.5, slot_aggregation='attentive-pooling', init_weight=0.2, **kargs):
    intent_embed = IntentEmbeddings(emb_size, intent_num, dropout=dropout)
    word_embed = WordEmbeddings(emb_size, vocab_size, pad_token_idxs['word'], dropout=dropout)
    slot_encoder = SlotEncoder(emb_size, hidden_size, num_layers, cell=cell, dropout=dropout, slot_aggregation=slot_aggregation)
    encoder = NLGEncoder(slot_encoder, hidden_size * 2, hidden_size, num_layers, cell=cell, dropout=dropout)
    enc2dec = StateTransition(hidden_size * 2, emb_size, hidden_size, num_layers, cell=cell, dropout=dropout)
    attn = Attention(hidden_size * 2, hidden_size, dropout=dropout, method='feedforward')
    decoder = NLGDecoderSCLSTMCopy(emb_size, hidden_size, emb_size, slot_num, num_layers, attn=attn, dropout=dropout)
    generator = NLGGeneratorCopy(hidden_size * 2 + hidden_size, hidden_size, vocab_size, slot_num=slot_num, dropout=dropout)
    m = SCLSTMCopyModel(intent_embed, word_embed, encoder, decoder, enc2dec, generator)
    if init_weight:
        for p in m.parameters():
            p.data.uniform_(- init_weight, init_weight)
        m.word_embed.embed.weight.data[pad_token_idxs['word']].zero_()
    return m