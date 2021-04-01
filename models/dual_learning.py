#coding=utf8
import os, sys
import torch
import torch.nn as nn
from utils.constants import PAD
from utils.example import tokenize_sent

class DualLearning(nn.Module):

    def __init__(self, slu_model, nlg_model, reward_model, slu_vocab, nlg_vocab, slu_evaluator, nlg_evaluator,
            alpha=0.5, beta=0.5, sample=5, slu_device='cpu', nlg_device='cpu', **kargs):
        """
        @args:
            1. alpha: coefficient for slu_reward = val_reward * alpha + rec_reward * (1 - alpha)
            2. beta: coefficient for nlg_reward = val_reward * beta + rec_reward * (1 - beta)
            3. sample: sampling size for training in dual learning cycles
        """
        super(DualLearning, self).__init__()
        self.slu_device, self.nlg_device = slu_device, nlg_device
        self.slu_model = slu_model.to(self.slu_device)
        self.nlg_model = nlg_model.to(self.nlg_device)
        self.reward_model = reward_model
        self.alpha, self.beta, self.sample = alpha, beta, sample
        self.slu_vocab, self.nlg_vocab = slu_vocab, nlg_vocab
        self.slu_evaluator, self.nlg_evaluator = slu_evaluator, nlg_evaluator

    def forward(self, *args, start_from='slu', **kargs):
        """
        @args:
            start_from: slu or nlg
        """
        if start_from == 'slu':
            return self.cycle_start_from_slu(*args, **kargs)
        else:
            return self.cycle_start_from_nlg(*args, **kargs)

    def cycle_start_from_slu(self, inputs, lens, raw_in):
        # start from slu model
        results = self.slu_model.decode_batch(inputs, lens, self.slu_vocab.slot2id, self.sample, self.sample)
        raw_in = [each for each in raw_in for _ in range(self.sample)] # repeat self.sample times
        results_dict = self.slu_evaluator.parse_outputs(results, words=raw_in, evaluation=False)
        raw_intents, raw_slots = results_dict['intents'], results_dict['slots']
        intent_scores, slot_scores = results['intent'][0], results['slot'][0]
        # compute validity reward, bsize x sample_size
        int_val_reward, slot_val_reward = self.reward_model(raw_slots, raw_intents, choice='slu_val')
        int_val_reward, slot_val_reward = int_val_reward.contiguous().view(-1, self.sample), slot_val_reward.contiguous().view(-1, self.sample)
        baseline = torch.mean(int_val_reward, dim=-1, keepdim=True)
        int_val_reward -= baseline
        baseline = torch.mean(slot_val_reward, dim=-1, keepdim=True)
        slot_val_reward -= baseline
        # forward into nlg model
        intents, slots, slot_lens, lens, slot_states, copy_tokens = self.slu2nlg(raw_slots, raw_intents)
        results = self.nlg_model.decode_batch(intents, slots, slot_lens, lens, slot_states, copy_tokens, self.nlg_vocab.word2id, self.sample, self.sample)
        raw_in = [each for each in raw_in for _ in range(self.sample)] # repeat self.sample times
        raw_slots = [each for each in raw_slots for _ in range(self.sample)] # repeat self.sample times
        results_dict = self.nlg_evaluator.parse_outputs(results, slots=raw_slots, evaluation=False)
        sentences, slot_acc = results_dict['sentences'], results_dict['slot_acc']
        nlg_scores = results['scores'].contiguous().view(-1, self.sample * self.sample)
        # compute reconstruction reward
        slot_acc = torch.tensor(slot_acc, dtype=torch.float)
        rec_reward = self.reward_model(sentences, raw_in, choice='slu_rec')
        slu_rec_reward = rec_reward.contiguous().view(-1, self.sample * self.sample)
        baseline = slu_rec_reward.mean(dim=-1, keepdim=True)
        slu_rec_reward = slu_rec_reward - baseline # do not use -=, will overwrite rec_reward
        slu_rec_reward = slu_rec_reward.contiguous().view(-1, self.sample, self.sample).mean(dim=-1)
        nlg_rec_reward = (0.5 * rec_reward + 0.5 * slot_acc).contiguous().view(-1, self.sample, self.sample)
        baseline = nlg_rec_reward.mean(dim=-1, keepdim=True)
        nlg_rec_reward -= baseline
        nlg_rec_reward = nlg_rec_reward.contiguous().view(-1, self.sample * self.sample)
        # calculate loss
        int_total_reward = self.alpha * int_val_reward + (1 - self.alpha) * slu_rec_reward
        slot_total_reward = self.alpha * slot_val_reward + (1 - self.alpha) * slu_rec_reward
        int_total_reward, slot_total_reward = int_total_reward.to(self.slu_device), slot_total_reward.to(self.slu_device)
        slu_loss = - torch.mean(0.5 * int_total_reward * intent_scores + 0.5 * slot_total_reward * slot_scores, dim=1)
        nlg_rec_reward = nlg_rec_reward.to(self.nlg_device)
        nlg_loss = - torch.mean((1 - self.alpha) * nlg_rec_reward * nlg_scores, dim=1)
        return slu_loss.sum(), nlg_loss.sum()

    def slu2nlg(self, slot_values, intents):
        vocab, device = self.nlg_vocab, self.nlg_device
        intents = [vocab.int2id[item] for item in intents]
        intents_tensor = torch.tensor(intents, dtype=torch.long, device=device)
        input_slots = [(list(sv), len(sv)) for ex in slot_values for sv in ex]
        if input_slots == []:
            slots_tensor, lens = None, None
        else:
            input_slots, lens = list(zip(*input_slots))
            max_len = max(lens)
            slot_idxs = [[vocab.word2id[item] for item in each] + [vocab.word2id[PAD]] * (max_len - len(each)) for each in input_slots]
            slots_tensor = torch.tensor(slot_idxs, dtype=torch.long, device=device)
            lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

        slot_lens = [len(ex) for ex in slot_values]
        max_slot_len = max(slot_lens)
        slot_lens_tensor = torch.tensor(slot_lens, dtype=torch.long, device=device)
        if max_slot_len == 0:
            copy_tokens = torch.zeros(slot_lens.size(0), 1, len(vocab.word2id), dtype=torch.float).to(device)
            slot_states = torch.zeros(slot_lens.size(0), len(vocab.slot2id), dtype=torch.float).to(device)
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
        return intents_tensor, slots_tensor, slot_lens_tensor, lens_tensor, slot_states, copy_tokens

    def cycle_start_from_nlg(self, intents, slots, slot_lens, lens, slot_states, copy_tokens, raw_intents, raw_slots):
        # start from nlg model
        results = self.nlg_model.decode_batch(intents, slots, slot_lens, lens, slot_states, copy_tokens, self.nlg_vocab.word2id, self.sample, self.sample)
        raw_intents = [each for each in raw_intents for _ in range(self.sample)] # repeat self.sample times
        raw_slots = [each for each in raw_slots for _ in range(self.sample)] # repeat self.sample times
        results_dict = self.nlg_evaluator.parse_outputs(results, slots=raw_slots, evaluation=False)
        surfaces, sentences, slot_acc = results_dict['surfaces'], results_dict['sentences'], results_dict['slot_acc']
        nlg_scores = results['scores']
        # compute validity reward, bsize x sample_size
        slot_acc = torch.tensor(slot_acc, dtype=torch.float)
        nlg_val_reward = self.reward_model(surfaces, sentences, choice='nlg_val')
        nlg_val_reward = (0.5 * nlg_val_reward + 0.5 * slot_acc).contiguous().view(-1, self.sample)
        baseline = nlg_val_reward.mean(dim=-1, keepdim=True)
        nlg_val_reward -= baseline
        # forward into slu model
        inputs, lens = self.nlg2slu(sentences)
        results = self.slu_model.decode_batch(inputs, lens, self.slu_vocab.slot2id, self.sample, self.sample)
        raw_intents = [each for each in raw_intents for _ in range(self.sample)]
        raw_slots = [each for each in raw_slots for _ in range(self.sample)]
        sentences = [each for each in sentences for _ in range(self.sample)]
        results_dict = self.slu_evaluator.parse_outputs(results, words=sentences, evaluation=False)
        pred_intents, pred_slots = results_dict['intents'], results_dict['slots']
        int_scores = results['intent'][0].contiguous().view(-1, self.sample * self.sample)
        slot_scores = results['slot'][0].contiguous().view(-1, self.sample * self.sample)
        # compute reconstruction reward
        int_rec_reward, slot_rec_reward = self.reward_model(pred_slots, pred_intents, raw_slots, raw_intents, choice='nlg_rec')
        nlg_rec_reward = (0.5 * int_rec_reward + 0.5 * slot_rec_reward).view(-1, self.sample * self.sample)
        baseline = nlg_rec_reward.mean(dim=-1, keepdim=True)
        nlg_rec_reward -= baseline
        nlg_rec_reward = nlg_rec_reward.contiguous().view(-1, self.sample, self.sample).mean(dim=-1)
        int_rec_reward = int_rec_reward.contiguous().view(-1, self.sample, self.sample)
        baseline = int_rec_reward.mean(dim=-1, keepdim=True)
        int_rec_reward -= baseline
        int_rec_reward = int_rec_reward.contiguous().view(-1, self.sample * self.sample)
        slot_rec_reward = slot_rec_reward.contiguous().view(-1, self.sample, self.sample)
        baseline = slot_rec_reward.mean(dim=-1, keepdim=True)
        slot_rec_reward -= baseline
        slot_rec_reward = slot_rec_reward.contiguous().view(-1, self.sample * self.sample)
        # calculate loss
        nlg_reward = self.beta * nlg_val_reward + (1 - self.beta) * nlg_rec_reward
        nlg_reward = nlg_reward.to(self.nlg_device)
        nlg_loss = - torch.mean(nlg_reward * nlg_scores, dim=1)
        int_rec_reward, slot_rec_reward = int_rec_reward.to(self.slu_device), slot_rec_reward.to(self.slu_device)
        slu_loss = 0.5 * int_rec_reward * int_scores + 0.5 * slot_rec_reward * slot_scores
        slu_loss = - torch.mean((1 - self.beta) * slu_loss, dim=1)
        return slu_loss.sum(), nlg_loss.sum()

    def nlg2slu(self, inputs):
        vocab, device = self.slu_vocab, self.slu_device
        lens = [len(ex) for ex in inputs]
        lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)
        max_len = max(lens)
        use_bert = hasattr(self.slu_model.word_embed, 'plm')
        if use_bert:
            subword_data = [tokenize_sent(words) for words in inputs]
            subword_inputs, subword_lens, subword_select_mask = list(zip(*subword_data))
            input_lens = [len(ex) for ex in subword_inputs]
            max_input_len = max(input_lens)
            subword_attention_mask = [[1] * len(ex) + [0] * (max_input_len - len(ex)) for ex in subword_inputs]
            subword_attention_mask_tensor = torch.tensor(subword_attention_mask, dtype=torch.bool, device=device)
            padded_subword_inputs = [ex + [0] * (max_input_len - len(ex)) for ex in subword_inputs]
            subword_inputs_tensor = torch.tensor(padded_subword_inputs, dtype=torch.long, device=device)
            subword_lens = [sl for subword_len in subword_lens for sl in subword_len]
            max_subword_len = max(subword_lens)
            subword_lens_tensor = torch.tensor(subword_lens, dtype=torch.long, device=device)
            subword_select_mask = [ex + [0] * (max_input_len - len(ex)) for ex in subword_select_mask]
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
        return inputs_tensor, lens_tensor

    def pad_embedding_grad_zero(self):
        self.slu_model.pad_embedding_grad_zero()
        self.nlg_model.pad_embedding_grad_zero()
