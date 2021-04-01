#coding=utf8
import torch
import numpy as np
from utils.constants import EMBEDDING, GK_EMB_DIM, PAD

def read_pretrained_vectors(filename, vocab, device):
    word2vec = {}
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line == '':
                continue
            word = line[:line.index(' ')]
            if word in vocab:
                values = line[line.index(' ') + 1:]
                word2vec[word] = torch.tensor(np.fromstring(values, sep=' ', dtype=np.float), device=device)
    return word2vec

def load_embeddings(dataset, module, word2id, device=None):
    emb_size = module.weight.data.size(-1)
    if emb_size != GK_EMB_DIM:
        print('Embedding size is not 400, randomly initialized')
        return 0.
    pretrained_vectors = read_pretrained_vectors(EMBEDDING(dataset), word2id, device)
    for word in pretrained_vectors:
        if word == PAD: continue
        module.weight.data[word2id[word]] = pretrained_vectors[word]
    return len(pretrained_vectors) / float(len(word2id))
