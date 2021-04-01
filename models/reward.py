#coding=utf8
import json, torch
import numpy as np
import editdistance as edt
from utils.constants import *
from utils.evaluator import calculate_bleu_score, calculate_slot_fscore

class RewardModel():

    def __init__(self, dataset, language_model, vocab, surface_level=False, device='cpu'):
        super(RewardModel, self).__init__()
        self.dataset = dataset
        self.language_model = language_model.to(device)
        self.vocab = vocab
        self.surface_level = surface_level
        self.intent_slots_dict = json.load(open(INTENT_SLOTS(self.dataset), 'r'))
        self.slot_values_dict = json.load(open(SLOT_VALUES(self.dataset), 'r'))
        self.noslot_prob = json.load(open(NOSLOT_PROB(self.dataset), 'r'))
        self.device = device

    def forward(self, *args, choice='slu_val'):
        if choice == 'slu_val':
            return self.slu_validity_reward(*args)
        elif choice == 'slu_rec':
            return self.slu_reconstruction_reward(*args)
        elif choice == 'nlg_val':
            return self.nlg_validity_reward(*args)
        else:
            return self.nlg_reconstruction_reward(*args)

    def slu_validity_reward(self, slots, intents):
        # calculate intent-slots/slot-values validity
        assert len(slots) == len(intents)
        intent_scores, slot_scores = [], []
        for i, s_v_list in zip(intents, slots):
            s_names = [s_v[0] for s_v in s_v_list]
            # intent-slots co-occurrence prob
            if i not in self.intent_slots_dict:
                # prevent intents not in train dataset
                intent_slots_score = 0.0
            elif len(s_names) == 0:
                # check whether allow empty slot for current intent
                intent_slots_score = 1.0 if self.noslot_prob[i] > 0 else 0.
            else:
                allowed_slots = self.intent_slots_dict[i]
                valid_slots = [x for x in s_names if x in allowed_slots]
                intent_slots_score = len(valid_slots) / float(len(s_names))
            intent_scores.append(intent_slots_score)
            # slot-values score
            if len(s_names) == 0:
                slot_values_score = 1.0
            else:
                s_values = [' '.join(s_v[2:]) for s_v in s_v_list]
                match_func = lambda x, y: max([1 - float(edt.eval(y.split(), ref.split())) / max([len(ref.split()), len(y.split())]) for ref in self.slot_values_dict[x]]) if x in self.slot_values_dict else .0
                unnormed_scores = list(map(match_func, s_names, s_values))
                slot_values_score = np.mean(unnormed_scores)
            slot_values_score = 0.5 * intent_slots_score + 0.5 * slot_values_score
            slot_scores.append(slot_values_score)
        intent_scores = torch.tensor(intent_scores, dtype=torch.float, requires_grad=False)
        slot_scores = torch.tensor(slot_scores, dtype=torch.float, requires_grad=False)
        return intent_scores, slot_scores

    def slu_reconstruction_reward(self, predictions, references):
        # calculate bleu score for each (pred, ref) tuple
        bleu_scores = [calculate_bleu_score(pred, [ref]) for pred, ref in zip(predictions, references)]
        return torch.tensor(bleu_scores, dtype=torch.float, requires_grad=False)

    def nlg_validity_reward(self, surfaces, sentences):
        # calculate language model length normalized log probability
        utterances = surfaces if self.surface_level else sentences
        vocab = self.vocab.sfm2id if self.surface_level else self.vocab.word2id
        input_idxs = [[vocab[BOS]] + [vocab[word] for word in sent] + [vocab[EOS]] for sent in utterances]
        lens = [len(each) for each in input_idxs]
        max_len = max(lens)
        input_idxs = [sent + [vocab[PAD]] * (max_len - len(sent)) for sent in input_idxs]
        input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.device)
        lens = torch.tensor(lens, dtype=torch.long, device=self.device)
        self.language_model.eval()
        with torch.no_grad():
            logprob = self.language_model.sent_logprob(input_tensor, lens, length_norm=True).cpu()
        return logprob

    def nlg_reconstruction_reward(self, pred_slots, pred_intents, raw_slots, raw_intents):
        # calculate similarity between predicted slots/intent and original slots/intent
        TP_FP_FN_list = [calculate_slot_fscore(pred, ref) for pred, ref in zip(pred_slots, raw_slots)]
        def calculate_f1score(tp_fp_fn):
            tp, fp, fn = tp_fp_fn
            if tp == 0:
                return 0.0
            f1 = 2 * float(tp) / (2 * tp + fp + fn)
            return f1
        slot_scores = list(map(calculate_f1score, TP_FP_FN_list))
        slot_scores = torch.tensor(slot_scores, dtype=torch.float, requires_grad=False)
        intent_scores = list(map(lambda x, y: y in x if type(x) in [list, tuple] else x == y, raw_intents, pred_intents))
        intent_scores = torch.tensor(intent_scores, dtype=torch.float, requires_grad=False)
        return intent_scores, slot_scores

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)
