#coding=utf8
import os, torch
import numpy as np
from utils.batch import get_minibatch
from utils.example import Example

def slu_decode(model, vocab, evaluator, data_inputs, output_path, test_batchSize, device='cpu', beam=5, n_best=1):
    use_bert = hasattr(model.word_embed, 'plm')
    data_index = np.arange(len(data_inputs))
    pred_slots, pred_intents = [], []
    ref_slots, ref_intents = [ex.label for ex in data_inputs], [ex.intents for ex in data_inputs]
    model.eval()
    for j in range(0, len(data_index), test_batchSize):
        inputs, _, _, lens, raw_in = get_minibatch(data_inputs, vocab, task='slu',
            data_index=data_index, index=j, batch_size=test_batchSize, device=device, use_bert=use_bert)
        with torch.no_grad():
            results = model.decode_batch(inputs, lens, vocab.slot2id, beam, n_best)
            outputs = evaluator.parse_outputs(results, words=raw_in, evaluation=True)
            pred_slots.extend(outputs['slots'])
            pred_intents.extend(outputs['intents'])
    (slot_fscore, overall_fscore), (intent_acc, overall_acc) = evaluator.compare_outputs(pred_slots, ref_slots, pred_intents, ref_intents)
    with open(output_path, 'w') as f:
        for idx, (sf, ia) in enumerate(zip(slot_fscore, intent_acc)):
            if sf < 1.0 or not ia:
                f.write('Input: ' + ' '.join(data_inputs[idx].words) + '\n')
                f.write('Ref slots: ' + ' , '.join([' '.join(s) for s in ref_slots[idx]]) + '\n')
                f.write('Pred slots: ' + ' , '.join([' '.join(s) for s in pred_slots[idx]]) + '\n')
                f.write('Ref intents: ' + ' , '.join(ref_intents[idx]) + '\n')
                f.write('Pred intents: ' + pred_intents[idx] + '\n\n')
        f.write('Overall slot fscore: %.4f ; intent accuracy: %.4f' % (overall_fscore, overall_acc))
    return overall_fscore, overall_acc

def slu_pseudo_labeling(model, vocab, evaluator, data_inputs, test_batchSize, device='cpu', beam=5):
    use_bert = hasattr(model.word_embed, 'plm')
    pseudo_labeled_dataset = []
    data_index = np.arange(len(data_inputs))
    model.eval()
    for j in range(0, len(data_index), test_batchSize):
        inputs, _, _, lens, raw_in = get_minibatch(data_inputs, vocab, task='slu',
            data_index=data_index, index=j, batch_size=test_batchSize, device=device, use_bert=use_bert)
        with torch.no_grad():
            results = model.decode_batch(inputs, lens, vocab.slot2id, beam, 1)
            results = evaluator.parse_outputs(results, evaluation=True)
            bio_list, intents = results['bio_list'], results['intents']
            pseudo_labeled_dataset.extend([Example(w, b, i) for w, b, i in zip(raw_in, bio_list, intents)])
    return pseudo_labeled_dataset

def nlg_decode(model, vocab, evaluator, data_inputs, output_path, test_batchSize, device='cpu', beam=5, n_best=1):
    data_index = np.arange(len(data_inputs))
    pred_surfaces, ref_surfaces = [], [ex.surface for ex in data_inputs]
    model.eval()
    for j in range(0, len(data_index), test_batchSize):
        intents, slots, slot_lens, lens, _, _, slot_states, copy_tokens, _ = get_minibatch(
            data_inputs, vocab, task='nlg', data_index=data_index, index=j, batch_size=test_batchSize, device=device)
        with torch.no_grad():
            results = model.decode_batch(intents, slots, slot_lens, lens, slot_states, copy_tokens, vocab.word2id, beam, n_best)
            surfaces = evaluator.parse_outputs(results, evaluation=True)['surfaces']
            pred_surfaces.extend(surfaces)
    (bleu_score, overall_score), (slot_acc, overall_acc) = evaluator.compare_outputs(pred_surfaces, ref_surfaces)
    with open(output_path, 'w') as f:
        for idx, (b, s) in enumerate(zip(bleu_score, slot_acc)):
            if b < overall_score or s < 1.0:
                f.write('Intent: ' + ' , '.join(data_inputs[idx].intents) + '\n')
                f.write('Slots: ' + ' , '.join([' '.join(s_v) for s_v in data_inputs[idx].slots]) + '\n')
                f.write('Ref: ' + ' '.join(ref_surfaces[idx]) + '\n')
                f.write('Pred: ' + ' '.join(pred_surfaces[idx]) + '\n\n')
        f.write('Overall bleu score and slot acc is: %.4f/%.4f' % (overall_score, overall_acc))
    return overall_score, overall_acc

def nlg_pseudo_labeling(model, vocab, evaluator, data_inputs, test_batchSize, device='cpu', beam=5):
    pseudo_labeled_dataset = []
    data_index = np.arange(len(data_inputs))
    model.eval()
    for j in range(0, len(data_index), test_batchSize):
        intents, slots, slot_lens, lens, _, _, slot_states, copy_tokens, (raw_intents, raw_slots) = get_minibatch(
            data_inputs, vocab, task='nlg', data_index=data_index, index=j, batch_size=test_batchSize, device=device)
        with torch.no_grad():
            results = model.decode_batch(intents, slots, slot_lens, lens, slot_states, copy_tokens, vocab.word2id, beam, 1)
            results = evaluator.parse_outputs(results, slots=raw_slots, evaluation=True)
            words, bio_list = results['sentences'], results['bio_list']
            pseudo_labeled_dataset.extend([Example(w, b, i) for w, b, i in zip(words, bio_list, raw_intents)])
    return pseudo_labeled_dataset

def lm_decode(model, vocab, evaluator, data_inputs, output_path, test_batchSize, device='cpu', surface_level=False):
    data_index = np.arange(len(data_inputs))
    ppl_list, logprob_list, lens_list, raw_inputs_list = [], [], [], []
    model.eval()
    for j in range(0, len(data_index), test_batchSize):
        inputs, lens, raw_inputs = get_minibatch(data_inputs, vocab, task='lm', data_index=data_index, index=j,
            batch_size=test_batchSize, device=device, surface_level=surface_level)
        with torch.no_grad():
            logprob = model.sent_logprob(inputs, lens)
            ppl = evaluator.parse_outputs(logprob, lens)
            lens_list.extend(lens.tolist())
            raw_inputs_list.extend(raw_inputs)
            logprob_list.extend(logprob.tolist())
            ppl_list.extend(ppl)
    overall_ppl = evaluator.compare_outputs(logprob_list, lens_list)
    with open(output_path, 'w') as f:
        for idx, (inputs, ppl) in enumerate(zip(raw_inputs_list, ppl_list)):
            if ppl >= overall_ppl:
                f.write('Inputs: ' + ' '.join(inputs) + '\n')
                f.write('Ppl: ' + str(ppl) + '\n\n')
        f.write('Overall ppl is: %.4f' % (overall_ppl))
    return overall_ppl