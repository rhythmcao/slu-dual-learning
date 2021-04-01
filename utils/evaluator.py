#coding=utf8
import os, sys, nltk
import numpy as np
from utils.example import bio2slots, bio2surface, surface2bio, Example
from utils.constants import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu_score(candidate_list, references_list, method=0, weights=(0.25, 0.25, 0.25, 0.25)):
    '''
    @args:
    if candidate_list is a single list, e.g. ['which','flight']
        references_list should be, e.g. [ ['which','flight'] , ['what','flight'] ]
        calculate bleu score of a single sentence
    if candidate_list is sentence list, e.g. [ ['which','flight'] , ['when','to','flight'] ]
        references_list should be, e.g.
        [ [ ['which','flight'] , ['what','flight'] ] , [ ['when','to','flight'] , ['when','to','go'] ] ]
        calculate the overall bleu score of multiple sentences, a whole corpus
    method(int): chencherry smoothing methods index
    '''
    chencherry = SmoothingFunction()
    if len(candidate_list) == 0:
        raise ValueError('[Error]: there is no candidate sentence!')
    if type(candidate_list[0]) == str:
        return sentence_bleu(references_list, candidate_list, weights,
                    eval('chencherry.method' + str(method)))
    else:
        return corpus_bleu(references_list, candidate_list, weights,
                    eval('chencherry.method' + str(method)))

def calculate_slot_fscore(prediction, reference):
    """ Single instance comparison
    @args:
        prediction: [(0, ${from_loc.city}, =, washington), (3, ${to_loc.city}, =, new)]
        reference: [(0, ${from_loc.city}, =, washington), (3, ${to_loc.city}, =, new, york)]
    @return:
        result: [TP_num, FP_num, FN_num]
    """
    TP, FP, FN = 0, 0, 0
    for slot_value in prediction:
        if slot_value in reference:
            TP += 1
        else:
            FP += 1
    for slot_value in reference:
        if slot_value not in prediction:
            FN += 1
    return TP, FP, FN

def calculate_slot_acc(prediction, reference):
    """
    @args:
        prediction: [${from_loc.city}, ${to_loc.city}]
        reference: [${from_loc.city}, ${to_loc.city}]
    @return:
        p, q, N: missing, redundant and total slot num
    """
    p, N = 0, len(reference)
    for s in reference:
        if s not in prediction:
            p += 1
        else: # remove the first occurrence
            prediction.remove(s)
    q = len(prediction)
    return p, q, N

class Evaluator():
    def __init__(self, vocab):
        super(Evaluator, self).__init__()
        self.vocab = vocab

    @classmethod
    def get_evaluator_from_task(cls, task, vocab):
        if task == 'slu':
            return SLUEvaluator(vocab)
        elif task == 'nlg':
            return NLGEvaluator(vocab)
        elif task == 'lm':
            return LMEvaluator(vocab)
        return Evaluator(vocab)

    def parse_outputs(self, *args, **kwargs):
        raise NotImplementedError

    def compare_outputs(self, *args, **kwargs):
        raise NotImplementedError

class SLUEvaluator(Evaluator):

    def parse_outputs(self, batch_outputs, words=None, evaluation=True):
        # evaluation: True, only parse the top of beam results
        _, slot_idx = batch_outputs['slot']
        _, intent_idx = batch_outputs['intent']
        id2slot, id2intent = self.vocab.id2slot, self.vocab.id2int
        return_dict = {'bio_list': [], 'slots': [], 'intents': []}
        if evaluation:
            bio_list = [[id2slot[idx] for idx in ex[0]] for ex in slot_idx]
            intents = [id2intent[ex[0]] for ex in intent_idx]
        else:
            bio_list = [[id2slot[idx] for idx in s] for ex in slot_idx for s in ex]
            intents = [id2intent[idx] for ex in intent_idx for idx in ex]
        return_dict['bio_list'], return_dict['intents'] = bio_list, intents
        if words is None:
            return return_dict
        slots = [bio2slots(words[idx], bio_list[idx], evaluation=evaluation) for idx in range(len(bio_list))]
        return_dict['slots'] = slots
        return return_dict

    def compare_outputs(self, slot_predictions, slot_references, intent_predictions, intent_references):
        """
        @args:
            slot_predictions: [ [(0, ${from_loc.city}, =, washington), (3, ${to_loc.city}, =, new)], ... ]
            slot_references: [ [(0, ${from_loc.city}, =, washington), (3, ${to_loc.city}, =, new, york)], ... ]
            intent_predictions: ['atis_meal', 'atis_meal', 'atis_flight']
            intent_references: ['atis_meal', 'atis_city', 'atis_flight'] or \
                [['atis_meal'], ['atis_city'], ['atis_flight', 'atis_city]]
        @return:
            slot_fscore: [fscore1, fscore2, ...]
            intent_accuracy: [True, False, ...]
        """
        slot_tp_fp_fn = list(map(lambda pred, ref: calculate_slot_fscore(pred, ref), slot_predictions, slot_references))
        slot_fscore = list(map(lambda item: 2 * float(item[0]) / (2 * item[0] + item[1] + item[2]) if sum(item) != 0 else 0., slot_tp_fp_fn))
        overall_tp_fp_fn = np.sum(slot_tp_fp_fn, axis=0)
        overall_fscore = 2 * float(overall_tp_fp_fn[0]) / (2 * overall_tp_fp_fn[0] + overall_tp_fp_fn[1] + overall_tp_fp_fn[2])
        intent_accuracy = list(map(lambda x, y: y in x if type(x) in [list, tuple] else x == y, intent_references, intent_predictions))
        overall_accuracy = np.mean(intent_accuracy, dtype=np.float)
        return (slot_fscore, overall_fscore), (intent_accuracy, overall_accuracy)

class NLGEvaluator(Evaluator):
    def parse_outputs(self, results, slots=None, evaluation=True):
        def trim(s, t):
            sentence = []
            for w in s:
                if w == t: break
                sentence.append(w)
            return sentence

        def filter_special(tok):
            return tok not in [PAD, BOS, EOS, EQUAL]
        if evaluation:
            predictions = [[self.vocab.id2word[tok] for tok in pred[0]] for pred in results['predictions']]
        else:
            predictions = [[self.vocab.id2word[tok] for tok in pred] for preds in results['predictions'] for pred in preds]
        trunc_predictions = [trim(ex, EOS) for ex in predictions]
        surfaces = [list(filter(filter_special, ex)) for ex in trunc_predictions]
        return_dict = {'surfaces': surfaces, 'sentences': [], 'bio_list': [], 'slot_acc': []}
        if slots is None: return return_dict
        else:
            sentences, bio_list, slot_acc = [], [], []
            for tmp_surface, tmp_slot in zip(surfaces, slots):
                tmp_sentence, tmp_bio_list, tmp_slot_acc = surface2bio(tmp_surface, tmp_slot)
                sentences.append(tmp_sentence)
                bio_list.append(tmp_bio_list)
                slot_acc.append(tmp_slot_acc)
            return_dict['sentences'], return_dict['bio_list'], return_dict['slot_acc'] = sentences, bio_list, slot_acc
        return return_dict

    def compare_outputs(self, predictions, references):
        """
        @args:
            predictions(list): list of predictions, e.g. [['when', 'to', 'fly'], ['which', 'flight'], ...]
            golden(list): list of golden utterances, e.g. [['when', 'fly'], ['which', 'flight'], ...]
        @return:
            bleu_score: list of bleu scores for each sample
            slot_acc: list of alignment scores for each sample, calculated by 1 - (p + q)/N
                p is number of missing slots, q is number of redundant slots, N is total number of slots in golden
        """
        surface_references = [[each] for each in references]
        bleu_score = list(map(calculate_bleu_score, predictions, surface_references))
        overall_bleu_score = calculate_bleu_score(predictions, surface_references)

        slot_predictions = [list(filter(lambda w: w.startswith('${'), each)) for each in predictions]
        slot_references = [list(filter(lambda w: w.startswith('${'), each)) for each in references]
        slot_p_q_n = list(map(calculate_slot_acc, slot_predictions, slot_references))
        slot_acc = list(map(lambda p_q_n: 1 - (p_q_n[0] + p_q_n[1]) / float(p_q_n[2]) if p_q_n[2] != 0 else 0., slot_p_q_n))
        overall_p_q_n = np.sum(slot_p_q_n, axis=0)
        overall_slot_acc = 1 - (overall_p_q_n[0] + overall_p_q_n[1] ) / float(overall_p_q_n[2])
        return (bleu_score, overall_bleu_score), (slot_acc, overall_slot_acc)

class LMEvaluator(Evaluator):
    def parse_outputs(self, logprob, lens):
        logprob, lens = logprob.tolist(), lens.tolist()
        ppl = [np.exp(- p / float(l - 1))  for p, l in zip(logprob, lens)]
        return ppl

    def compare_outputs(self, logprob_list, lens_list):
        logprob = sum(logprob_list)
        lens = sum(lens_list) - len(lens_list)
        ppl = np.exp(- logprob / float(lens))
        return ppl
