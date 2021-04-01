"""Read SLU Vocabulary"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
from collections import defaultdict

class Vocab():

    def __init__(self, dataset='atis', task='slu'):
        super(Vocab, self).__init__()
        self.dataset = dataset
        if task == 'slu':
            self.word2id, self.id2word = self.read_vocab_file(WORDVOCAB(dataset), padding=True, boundary=True, unk=True)
            self.slot2id, self.id2slot = self.read_vocab_file(BIOVOCAB(dataset), padding=True, boundary=True, unk=False)
            self.int2id, self.id2int = self.read_vocab_file(INTENTVOCAB(dataset), padding=False, boundary=False, unk=False)
        elif task == 'nlg':
            self.word2id, self.id2word = self.read_mixed_vocab_file(SLOTVOCAB(dataset), WORDVOCAB(dataset), padding=True, boundary=True, unk=True, specials=[EQUAL])
            self.slot2id, self.id2slot = self.read_vocab_file(SLOTVOCAB(dataset), padding=False, boundary=False, unk=False)
            self.int2id, self.id2int = self.read_vocab_file(INTENTVOCAB(dataset), padding=False, boundary=False, unk=False)
        elif task == 'lm':
            # surface form~(sfm) vocab
            self.sfm2id, self.id2sfm = self.read_mixed_vocab_file(SLOTVOCAB(dataset), WORDVOCAB(dataset), padding=True, boundary=True, unk=True, specials=[])
            self.word2id, self.id2word = self.read_vocab_file(WORDVOCAB(dataset), padding=True, boundary=True, unk=True)

    def read_mixed_vocab_file(self, path1, path2, padding=True, boundary=True, unk=True, specials=[]):
        # the first part is slots, then followed by words and special symbol EQUAL
        word2id, id2word = {}, {}
        with open(path1, 'r') as f:
            for line in f:
                word = line.strip()
                if word == '' or word in word2id: continue
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word
        if padding:
            word2id[PAD] = len(word2id)
            id2word[len(id2word)] = PAD
        if boundary:
            word2id[BOS] = len(word2id)
            id2word[len(id2word)] = BOS
            word2id[EOS] = len(word2id)
            id2word[len(id2word)] = EOS
        if unk:
            word2id[UNK] = len(word2id)
            id2word[len(id2word)] = UNK
        with open(path2, 'r') as f:
            for line in f:
                word = line.strip()
                if word == '' or word in word2id: continue
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word
        for word in specials:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word
        return word2id, id2word

    def read_vocab_file(self, vocab_path, padding=True, boundary=False, unk=True):
        word2id, id2word = {}, {}
        if padding: # always the first symbol
            word2id[PAD] = len(word2id)
            id2word[len(id2word)] = PAD
        if boundary:
            word2id[BOS] = len(word2id)
            id2word[len(id2word)] = BOS
            word2id[EOS] = len(word2id)
            id2word[len(id2word)] = EOS
        if unk:
            word2id[UNK] = len(word2id)
            id2word[len(id2word)] = UNK
        with open(vocab_path, 'r') as f:
            for line in f:
                word = line.strip()
                if word == '' or word in word2id: continue
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word
        return word2id, id2word
