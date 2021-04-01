#coding=utf8
import os, sys, argparse, json, operator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
from utils.dataset import read_dataset
from collections import defaultdict
from embeddings import GloveEmbedding, KazumaCharEmbedding

def construct_database_and_com(dataset, ex_list):
    slot_values_dict, intent_slots_dict = defaultdict(set), defaultdict(set)
    noslot_prob = defaultdict(lambda : [0, 0])
    for ex in ex_list:
        intent = ex.intents
        for i in intent:
            noslot_prob[i][0] += 1
            if len(ex.slots) == 0:
                noslot_prob[i][1] += 1
            for s_v in ex.slots:
                s, v = s_v[0], ' '.join(s_v[2:])
                slot_values_dict[s].add(v)
                intent_slots_dict[i].add(s)
    for each in slot_values_dict:
        slot_values_dict[each] = sorted(list(slot_values_dict[each]))
    for each in intent_slots_dict:
        intent_slots_dict[each] = sorted(list(intent_slots_dict[each]))
    for each in noslot_prob:
        noslot_prob[each] = noslot_prob[each][1] / float(noslot_prob[each][0])
    json.dump(slot_values_dict, open(SLOT_VALUES(dataset), 'w'), indent=4)
    json.dump(intent_slots_dict, open(INTENT_SLOTS(dataset), 'w'), indent=4)
    json.dump(noslot_prob, open(NOSLOT_PROB(dataset), 'w'), indent=4)

def construct_vocab(dataset, ex_list, mwf=2):
    word_vocab, bio_vocab, slot_vocab, intent_vocab = defaultdict(lambda: 0), set(), set(), set()
    for ex in ex_list:
        words, bios, intents = ex.words, ex.bios, ex.intents
        for w in words:
            word_vocab[w] += 1
        bio_vocab.update(bios)
        intent_vocab.update(intents)

    sorted_words = sorted(word_vocab.items(), key=operator.itemgetter(1), reverse=True)
    words = [x[0] for x in sorted_words if x[1] >= mwf]
    with open(WORDVOCAB(dataset), 'w') as of:
        for each in words:
            of.write(each + '\n')
    print('Size of word vocabulary in %s is %d' % (dataset, len(words)))

    bios = sorted(list(bio_vocab))
    with open(BIOVOCAB(dataset), 'w') as of:
        for each in bios:
            of.write(each + '\n')
    print('Size of BIO label vocabulary in %s is %d' % (dataset, len(bios)))

    for each in bios:
        if each != 'O':
            slot_vocab.add('${' + each.lstrip('B-').lstrip('I-') + '}')
    slots = sorted(list(slot_vocab))
    with open(SLOTVOCAB(dataset), 'w') as of:
        for each in slots:
            of.write(each + '\n')
    print('Size of slot vocabulary in %s is %d' % (dataset, len(slots)))

    intents = sorted(list(intent_vocab))
    with open(INTENTVOCAB(dataset), 'w') as of:
        for each in intents:
            of.write(each + '\n')
    print('Size of intent vocabulary in %s is %d' % (dataset, len(intents)))
    return words, bios, slots, intents

def get_pretrained_embeddings(dataset, words, slots, intents):
    vocab = set(words + slots + intents)
    for symbol in [BOS, EOS, UNK, EQUAL]:
        vocab.add(symbol)

    # GK Embedding
    word_embed, char_embed = GloveEmbedding(default='zero'), KazumaCharEmbedding()
    embed_size = word_embed.d_emb + char_embed.d_emb
    progress = 0
    with open(EMBEDDING(dataset), 'w') as out_file:
        for word in vocab:
            progress += 1
            vector = word_embed.emb(word) + char_embed.emb(word)
            string = ' '.join([str(v) for v in vector])
            out_file.write(word + ' ' + string + '\n')
            if progress % 1000 == 0:
                print("Retrieve 400-dim GK Embedding for the", progress, "-th word ...")
    print('In total, process %d words in %s' % (len(vocab), dataset))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, nargs='+')
    parser.add_argument('--mwf', type=int, default=1, help='minimum word frequency')
    args = parser.parse_args(sys.argv[1:])

    for d in args.dataset:
        print('\nStart processing domain %s ...' % (d))
        ex_list = read_dataset(d, 'train') + read_dataset(d, 'valid') + read_dataset(d, 'test')
        words, bios, slots, intents = construct_vocab(d, ex_list, args.mwf)
        construct_database_and_com(d, ex_list)
        get_pretrained_embeddings(d, words, slots, intents)
