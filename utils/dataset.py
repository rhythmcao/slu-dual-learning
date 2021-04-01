#coding=utf8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import DATAPATH
from utils.example import *
import numpy as np

def read_dataset(dataset='atis', choice='train'):
    assert choice in ['train', 'valid', 'test']
    assert dataset in  ['atis', 'snips']
    filepath = DATAPATH(dataset, choice)

    def read_dataset_from_filepath(filepath):
        dataset = []
        with open(filepath, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                sentence, intents = line.split(' <=> ')
                chunks = map(lambda item: item.split(':'), filter(lambda item: ':' in item, sentence.split(' ')))
                words, bios = zip(*map(lambda item: (':'.join(item[:-1]), item[-1]), chunks))
                words = map(lambda item: '_' if item == '' else item, words)
                dataset.append(Example(words, bios, intents))
        return dataset

    return read_dataset_from_filepath(filepath)

def split_dataset(dataset, split_ratio=1.0):
    split_seed = 999
    assert split_ratio >= 0. and split_ratio <= 1.0
    index = np.arange(len(dataset))
    state = np.random.get_state()
    np.random.seed(split_seed)
    np.random.shuffle(index)
    np.random.set_state(state)
    splt = int(len(dataset) * split_ratio)
    first = [dataset[idx] for idx in index[:splt]]
    second = [dataset[idx] for idx in index[splt:]]
    return first, second