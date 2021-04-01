#coding=utf8
import os

PAD = '[PAD]'
UNK = '[UNK]'
BOS = '[CLS]'
EOS = '[SEP]'
EQUAL = '='
DATAPATH = lambda d, s: os.path.join('data', d, s)
GK_EMB_DIM = 400
EMBEDDING = lambda d: os.path.join('data/.cache/%s_gk_400.txt' % (d))
WORDVOCAB = lambda d: os.path.join('data', d, 'vocab.word')
BIOVOCAB = lambda d: os.path.join('data', d, 'vocab.bio')
SLOTVOCAB = lambda d: os.path.join('data', d, 'vocab.slot')
INTENTVOCAB = lambda d: os.path.join('data', d, 'vocab.intent')
DECODE_MAX_LENGTH = 50
SLOT_VALUES = lambda d: os.path.join('data', d, 'slot_values.json')
INTENT_SLOTS = lambda d: os.path.join('data', d, 'intent_slots.json')
NOSLOT_PROB = lambda d: os.path.join('data', d, 'noslot_prob.json')
