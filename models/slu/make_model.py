#coding=utf8
from models.slu.embedding import SLUEmbeddings, SLUPretrainedEmbedding
from models.slu.encoder import RNNEncoder
from models.slu.init_decoder import StateTransition
from models.slu.crf import CRF
from models.slu.slot_decoder import SlotDecoder, SlotDecoderFocus
from models.slu.intent_decoder import IntentDecoder
from models.slu.slu_model_birnn import BiRNNModel
from models.slu.slu_model_birnn_crf import BiRNNCRFModel
from models.slu.slu_model_focus import FocusModel

def construct_models(**kwargs):
    model_type = kwargs.pop('model_type', 'focus')
    constructor = {
        'birnn': construct_model_birnn,
        'birnn+crf': construct_model_birnn_crf,
        'focus': construct_model_focus
    }
    return constructor[model_type](**kwargs)

def construct_model_birnn(emb_size=400, vocab_size=952, pad_token_idxs={"word": 0, "slot": 0},
        slot_num=130, intent_num=18, hidden_size=256, num_layers=1, cell='lstm', intent_method='hiddenAttn',
        use_bert=False, subword_aggregation='attentive-pooling', dropout=0.5, init_weight=0.2, lazy_load=False, **kargs):
    word_embed = SLUPretrainedEmbedding(subword_aggregation=subword_aggregation, lazy_load=lazy_load) if use_bert \
        else SLUEmbeddings(emb_size, vocab_size, pad_token_idxs["word"], dropout=dropout)
    encoder = RNNEncoder(word_embed.emb_size, hidden_size, num_layers, cell, dropout=dropout)
    slot_decoder = SlotDecoder(hidden_size, slot_num, dropout=dropout, log_prob=True)
    intent_decoder = IntentDecoder(hidden_size, intent_num, cell, dropout=dropout, method=intent_method)
    m = BiRNNModel(word_embed, encoder, slot_decoder, intent_decoder)
    if init_weight:
        for n, p in m.named_parameters():
            if 'plm' not in n:
                p.data.uniform_(- init_weight, init_weight)
        if not use_bert:
            m.word_embed.embed.weight.data[pad_token_idxs['word']].zero_()
    return m

def construct_model_birnn_crf(emb_size=400, vocab_size=952, pad_token_idxs={"word": 0, "slot": 0}, start_idx=-2, end_idx=-1,
        slot_num=130, intent_num=18, hidden_size=256, num_layers=1, cell='lstm', intent_method='hiddenAttn',
        use_bert=False, subword_aggregation='attentive-pooling', dropout=0.5, init_weight=0.2, lazy_load=False, **kargs):
    word_embed = SLUPretrainedEmbedding(subword_aggregation=subword_aggregation, lazy_load=lazy_load) if use_bert \
        else SLUEmbeddings(emb_size, vocab_size, pad_token_idxs["word"], dropout=dropout)
    encoder = RNNEncoder(word_embed.emb_size, hidden_size, num_layers, cell, dropout=dropout)
    slot_decoder = SlotDecoder(hidden_size, slot_num, dropout=dropout, log_prob=False)
    crf_layer = CRF(slot_num, start_idx=start_idx, end_idx=end_idx)
    intent_decoder = IntentDecoder(hidden_size, intent_num, cell, dropout=dropout, method=intent_method)
    m = BiRNNCRFModel(word_embed, encoder, slot_decoder, crf_layer, intent_decoder)
    if init_weight:
        for n, p in m.named_parameters():
            if 'plm' not in n and 'crf_layer' not in n:
                p.data.uniform_(- init_weight, init_weight)
        if not use_bert:
            m.word_embed.embed.weight.data[pad_token_idxs['word']].zero_()
    return m

def construct_model_focus(emb_size=400, vocab_size=952, pad_token_idxs={"word": 0, "slot": 0},
        slot_num=130, intent_num=18, hidden_size=256, num_layers=1, cell='lstm', intent_method='hiddenAttn',
        use_bert=False, subword_aggregation='attentive-pooling', dropout=0.5, init_weight=0.2, lazy_load=False, **kargs):
    word_embed = SLUPretrainedEmbedding(subword_aggregation=subword_aggregation, lazy_load=lazy_load) if use_bert \
        else SLUEmbeddings(emb_size, vocab_size, pad_token_idxs["word"], dropout=dropout)
    encoder = RNNEncoder(word_embed.emb_size, hidden_size, num_layers, cell, dropout=dropout)
    enc2dec = StateTransition(num_layers, cell)
    slot_embed = SLUEmbeddings(emb_size, slot_num, pad_token_idxs["slot"], dropout=dropout)
    slot_decoder = SlotDecoderFocus(emb_size, hidden_size, slot_num, num_layers=num_layers, cell=cell, dropout=dropout)
    intent_decoder = IntentDecoder(hidden_size, intent_num, cell, dropout=dropout, method=intent_method)
    m = FocusModel(word_embed, encoder, enc2dec, slot_embed, slot_decoder, intent_decoder)
    if init_weight:
        for n, p in m.named_parameters():
            if 'plm' not in n:
                p.data.uniform_(- init_weight, init_weight)
        if not use_bert:
            m.word_embed.embed.weight.data[pad_token_idxs["word"]].zero_()
        m.slot_embed.embed.weight.data[pad_token_idxs["slot"]].zero_()
    return m
