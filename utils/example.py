#coding=utf8
import random
from utils.constants import *
from collections import defaultdict
from transformers import AutoTokenizer

def bio2slots(word_list, slot_list, evaluation=False):
    """
    @args:
        word_list: ['list', 'flights', 'from', 'washington', 'to', 'new', 'york']
        slot_list: ['O', 'O', 'O', 'B-from_loc.city', 'O', 'B-to_loc.city', I-to_loc.city']
        evaluation:
            True: reserve start index and allow duplicate (slot, value) pair
            False: ignore start index and remove duplicate (slot, value) pair
    @return:
        slots:
            evaluation is True: [('3', '${from_loc.city}', '=', 'washington'), ('5', '${to_loc.city}', '=', 'new', 'york')]
            evaluation is False: [('${from_loc.city}', '=', 'washington'), ('${to_loc.city}', '=', 'new', 'york')]
    """
    word_list = word_list[:word_list.index(PAD)] if PAD in word_list else word_list
    prev, prev_chunk, slots = 'O', [], []
    for i, (w, s) in enumerate(zip(word_list, slot_list)):
        if s.startswith('B-'):
            if prev != 'O':
                slots.append(prev_chunk)
            prev, prev_chunk = s.lstrip('B-'), [str(i), '${' + s.lstrip('B-') + '}', EQUAL, w]
        elif s.startswith('I-'):
            if prev == s.lstrip('I-'):
                prev_chunk.append(w)
            elif prev == 'O': # should startswith B-
                prev, prev_chunk = s.lstrip('I-'), [str(i), '${' + s.lstrip('I-') + '}', EQUAL, w]
            else: # prev slot is not the same slot
                slots.append(prev_chunk)
                prev, prev_chunk = s.lstrip('I-'), [str(i), '${' + s.lstrip('I-') + '}', EQUAL, w]
        else:
            if prev != 'O':
                slots.append(prev_chunk)
            prev, prev_chunk = 'O', []
    if prev != 'O':
        slots.append(prev_chunk)
    if not evaluation:
        before_set = [' '.join(s_v[1:]) for s_v in slots]
        after_set = sorted(set(before_set), key=before_set.index)
        return [each.split(' ') for each in after_set]
    else:
        return slots

def bio2surface(word_list, slot_list):
    """
    @args:
        word_list: ['list', 'flights', 'from', 'washington', 'to', 'new', 'york']
        slot_list: ['O', 'O', 'O', 'B-from_loc.city', 'O', 'B-to_loc.city', I-to_loc.city']
    @return:
        ['list', 'flights', 'from', '${from_loc.city}', 'to', '${to_loc.city}']
    """
    # assert len(word_list) == len(slot_list)
    word_list = [each[: each.index(PAD)] if PAD in each else each for each in word_list]
    tuples = list(zip(word_list, slot_list))
    prev, surface = 'O', []
    for w, s in tuples:
        if s.startswith('B-'):
            if prev != 'O':
                surface.append('${' + prev +'}')
            prev = s.lstrip('B-')
        elif s.startswith('I-'):
            if prev == s.lstrip('I-') or prev == 'O':
                pass
            else: # prev slot is not the same slot
                surface.append('${' + prev + '}')
            prev = s.lstrip('I-')
        else:
            if prev != 'O':
                surface.append('${' + prev +'}')
            surface.append(w)
            prev = 'O'
    if prev != 'O':
        surface.append('${' + prev +'}')
    return surface

def surface2bio(surface, slots):
    """
    @args:
        surface: ['list', 'flight', 'from', '${from_loc.city}', 'to', '${to_loc.city}']
        slots: [['${from_loc.city}', '=', 'washington'], ['${to_loc.city}', '=', 'new', 'york']]
    @return:
        sentence: ['list', 'flight', 'from', 'washington', 'to', 'new', 'york']
        bio_list: ['O', 'O', 'O', 'B-from_loc.city', 'to', 'B-to_loc.city', 'I-to_loc.city']
        score: reward about the predicted surface,
            err_cnt = 1 * num_slots_not_provided + 0.5 * num_redundant_slots + 1.0 * num_missing_slots
            score = 1 - err_cnt / num_slots
    """
    sentence, bio_list = [], []
    slot_values, counter = defaultdict(list), defaultdict(lambda : 0)
    for s_v in slots:
        slot_values[s_v[0]].append(list(s_v[2:]))
    total_num = sum(map(lambda k: len(slot_values[k]), slot_values))
    err_cnt = 0
    for each in surface:
        if not each.startswith('${'):
            sentence.append(each)
            bio_list.append('O')
        elif each not in slot_values:
            # raise ValueError('Slot name %s should not appear in the prediction: %s' % (each, ' '.join(surface)))
            err_cnt += 1 # penalize for slots not provided
            continue
        elif len(slot_values[each]) > counter[each]:
            # extract the next slot value sequentially
            value = slot_values[each][counter[each]]
            counter[each] += 1
            sentence.extend(value)
            s = each.lstrip('${').rstrip('}')
            label = ['B-' + s] + ['I-' + s] * (len(value) - 1)
            bio_list.extend(label)
        else: # all slot values are used
            err_cnt += 0.5 # penalize for redundant slots
            value = random.choice(slot_values[each])
            sentence.extend(value)
            s = each.lstrip('${').rstrip('}')
            label = ['B-' + s] + ['I-' + s] * (len(value) - 1)
            bio_list.extend(label)
    for each in slot_values: # penalize for slots not used
        err_cnt += len(slot_values[each]) - counter[each]
    score = 1 - err_cnt / float(total_num) if total_num != 0 else 1.0
    return sentence, bio_list, score

class Example():

    tokenizer = AutoTokenizer.from_pretrained('./data/.cache/bert-base-uncased')

    def __init__(self, words, bios, intents, conf=1.0):
        super(Example, self).__init__()
        self.words, self.bios = list(words), list(bios)
        self.slots = bio2slots(self.words, self.bios)
        self.surface = bio2surface(self.words, self.bios)
        self.intents = intents.split(';')
        # label record position info, used during slu evaluation
        self.label = bio2slots(self.words, self.bios, evaluation=True)
        self.conf = conf # confidence score (pseudo labeling method)

        self.subword_id, self.subword_len, self.subword_select_mask = tokenize_sent(self.words)

def tokenize_sent(words):
    t = Example.tokenizer
    subword_id, subword_len = [t.cls_token_id], []
    for w in words:
        toks = t.convert_tokens_to_ids(t.tokenize(w))
        subword_id.extend(toks)
        subword_len.append(len(toks))
    subword_id.append(t.sep_token_id)
    subword_select_mask = [0] + [1] * (len(subword_id) - 2) + [0] # remove [CLS] and [SEP]
    return subword_id, subword_len, subword_select_mask