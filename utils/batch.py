#coding=utf8
import sys, os, random
import torch
from utils.constants import *

def get_minibatch(data_list, vocab, task='slu', data_index=None, index=0, batch_size=16, device=None, **kargs):
    index = index % len(data_list)
    batch_data_list = [data_list[idx] for idx in data_index[index: index + batch_size]]
    return BATCH_FUNC[task](batch_data_list, vocab, device, **kargs)

def get_minibatch_slu(ex_list, vocab, device, use_bert=False, **kargs):
    inputs = [ex.words for ex in ex_list]
    lens = [len(ex) for ex in inputs]
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)
    max_len = max(lens)

    if use_bert:
        subword_inputs = [ex.subword_id for ex in ex_list]
        input_lens = [len(ex) for ex in subword_inputs]
        max_input_len = max(input_lens)
        subword_attention_mask = [[1] * len(ex) + [0] * (max_input_len - len(ex)) for ex in subword_inputs]
        subword_attention_mask_tensor = torch.tensor(subword_attention_mask, dtype=torch.bool, device=device)
        padded_subword_inputs = [ex + [0] * (max_input_len - len(ex)) for ex in subword_inputs]
        subword_inputs_tensor = torch.tensor(padded_subword_inputs, dtype=torch.long, device=device)

        subword_lens = [sl for ex in ex_list for sl in ex.subword_len]
        max_subword_len = max(subword_lens)
        subword_lens_tensor = torch.tensor(subword_lens, dtype=torch.long, device=device)
        subword_select_mask = [ex.subword_select_mask + [0] * (max_input_len - len(ex.subword_select_mask)) for ex in ex_list]
        subword_select_mask_tensor = torch.tensor(subword_select_mask, dtype=torch.bool, device=device)
        inputs_tensor = {
            'inputs': {
                'input_ids': subword_inputs_tensor,
                'attention_mask': subword_attention_mask_tensor
            },
            'lens': lens_tensor,
            'subword_select_mask': subword_select_mask_tensor,
            'subword_lens': subword_lens_tensor
        }
    else:
        padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in inputs]
        inputs_idx = [[vocab.word2id[w] for w in sent] for sent in padded_inputs]
        inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)

    outputs = [ex.bios for ex in ex_list]
    padded_outputs = [[BOS] + sent + [PAD] * (max_len - len(sent)) for sent in outputs]
    outputs_idx = [[vocab.slot2id[w] for w in sent] for sent in padded_outputs]
    outputs_tensor = torch.tensor(outputs_idx, dtype=torch.long, device=device)

    raw_intents = [ex.intents for ex in ex_list]
    intents = [vocab.int2id[random.choice(item)] for item in raw_intents]
    intents_tensor = torch.tensor(intents, dtype=torch.long, device=device)
    return inputs_tensor, outputs_tensor, intents_tensor, lens_tensor, inputs

def get_minibatch_nlg(ex_list, vocab, device, **kargs):
    input_intents = [random.choice(ex.intents) for ex in ex_list]
    intents = [vocab.int2id[item] for item in input_intents]
    intents_tensor = torch.tensor(intents, dtype=torch.long, device=device)

    slot_values = [ex.slots for ex in ex_list]
    input_slots = [(list(sv), len(sv)) for ex in slot_values for sv in ex]
    if input_slots == []:
        lens, slots_tensor = None, None
    else:
        input_slots, lens = list(zip(*input_slots))
        max_len = max(lens)
        slot_idxs = [[vocab.word2id[item] for item in each] + [vocab.word2id[PAD]] * (max_len - len(each)) for each in input_slots]
        slots_tensor = torch.tensor(slot_idxs, dtype=torch.long, device=device)
        lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

    raw_outputs = [ex.words for ex in ex_list]
    surface = [ex.surface for ex in ex_list]
    outputs = [[BOS] + each + [EOS] for each in surface]
    out_lens = [len(each) for each in outputs]
    max_out_len = max(out_lens)
    out_lens_tensor = torch.tensor(out_lens, dtype=torch.long, device=device)
    outputs = [[vocab.word2id[item] for item in each] + [vocab.word2id[PAD]] * (max_out_len - len(each)) for each in outputs]
    outputs_tensor = torch.tensor(outputs, dtype=torch.long, device=device)

    slot_lens = [len(ex.slots) for ex in ex_list]
    max_slot_len = max(slot_lens)
    slot_lens_tensor = torch.tensor(slot_lens, dtype=torch.long, device=device)
    if max_slot_len == 0:
        copy_tokens = torch.zeros(slot_lens_tensor.size(0), 1, len(vocab.word2id), dtype=torch.float).to(device)
        slot_states = torch.zeros(slot_lens_tensor.size(0), len(vocab.slot2id), dtype=torch.float).to(device)
    else:
        slot_states_idxs = [[vocab.word2id[s_v[0]] for s_v in ex] for ex in slot_values]
        copy_tokens = [
            torch.cat([
                torch.zeros(len(each), len(vocab.word2id), dtype=torch.float)\
                    .scatter_(-1, torch.tensor(each, dtype=torch.long).unsqueeze(-1), 1.0),
                torch.zeros(max_slot_len - len(each), len(vocab.word2id), dtype=torch.float)
            ], dim=0)
            if each != [] else
                torch.zeros(max_slot_len, len(vocab.word2id), dtype=torch.float)
            for each in slot_states_idxs
        ]
        copy_tokens = torch.stack(copy_tokens, dim=0).to(device)
        slot_states = torch.sum(copy_tokens, dim=1)[:, :len(vocab.slot2id)]
    return intents_tensor, slots_tensor, slot_lens_tensor, lens_tensor, outputs_tensor, out_lens_tensor, slot_states, copy_tokens, \
        (input_intents, slot_values)

def get_minibatch_lm(ex_list, vocab, device, surface_level=False, **kargs):
    vocab = vocab.sfm2id if surface_level else vocab.word2id
    raw_inputs = [[BOS] + ex.surface + [EOS] for ex in ex_list] if surface_level else [[BOS] + ex.words + [EOS] for ex in ex_list]
    inputs = [[vocab[word] for word in each] for each in raw_inputs]
    lens = [len(each) for each in inputs]
    max_len = max(lens)
    inputs = [each + [vocab[PAD]] * (max_len - len(each)) for each in inputs]
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)
    return inputs_tensor, lens_tensor, raw_inputs

BATCH_FUNC = {
    "slu": get_minibatch_slu,
    "nlg": get_minibatch_nlg,
    "lm": get_minibatch_lm,
}
