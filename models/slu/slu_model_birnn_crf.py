#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slu.Beam import Beam
from models.slu.crf import CRF
from models.model_utils import lens2mask

class BiRNNCRFModel(nn.Module):

    def __init__(self, word_embed, encoder, slot_decoder, crf_layer, intent_decoder):
        super(BiRNNCRFModel, self).__init__()
        self.word_embed = word_embed
        self.encoder = encoder
        self.slot_decoder = slot_decoder
        self.crf_layer = crf_layer
        self.intent_decoder = intent_decoder

    def pad_embedding_grad_zero(self):
        self.word_embed.pad_embedding_grad_zero()

    def forward(self, inputs, lens, dec_inputs=None):
        outputs, hidden_states = self.encoder(self.word_embed(inputs), lens)
        bios = self.slot_decoder(outputs)
        bios = self.crf_layer.neg_log_likelihood_loss(bios, lens2mask(lens), dec_inputs[:, 1:])
        intents = self.intent_decoder(hidden_states, outputs, lens)
        return bios, intents

    def decode_batch(self, inputs, lens, vocab, beam=5, n_best=1, **kargs):
        outputs, hidden_states = self.encoder(self.word_embed(inputs), lens)
        bios = self.slot_decoder(outputs)
        intents = self.intent_decoder(hidden_states, outputs, lens)
        if n_best == 1:
            return self.decode_greed(bios, intents, lens)
        else:
            return self.decode_beam_search(bios, intents, lens, n_best, **kargs)

    def decode_greed(self, bios, intents, lens):
        """
        @args:
            bios(torch.FloatTensor): bsize x seqlen x slot_num
            intents(torch.FloatTensor): bsize x intent_num
            lens(torch.LongTensor): bsize
        @return:
            dict:
                slot: tuple of
                    slot_score(torch.FloatTensor): bsize x 1
                    slot_idx(list): such as [ [[1, 2, 4, 9, 11]] , [[1, 2, 5, 8, 10]] , ... ]
                intent: tuple of
                    intent_score(torch.FloatTensor): bsize x 1
                    intent_idx(list): bsize x 1
        """
        slot_score, slot_idx = self.crf_layer._viterbi_decode(bios, lens2mask(lens))
        int_score, int_idx = torch.max(intents, dim=1, keepdim=True)
        return {"intent": (int_score, int_idx.tolist()), "slot": (slot_score, slot_idx.tolist())}

    def decode_beam_search(self, bios, intents, lens, n_best=1, **kargs):
        """
        @args:
            n_best(int): number of predictions to return
        @return:
            dict:
            (key)intent: (value) tuple of (n_best most likely intent score, n_best most likely intent idx)
                intent_score(torch.FloatTensor): bsize x n_best
                intent_idx(list): bsize x n_best
            (key)slot: (value) tuple of (n_best most likely seq score, n_best most likely seq idx)
                slot_score(torch.FloatTensor): bsize x n_best
                slot_idx(list): n_best=2, such as [ [[1, 2, 4, 9], [1, 2, 3, 5]] , [[1, 2, 5], [1, 2, 3]] , ... ]
        """
        slot_scores, slot_idxs = self.crf_layer._viterbi_decode_nbest(bios, lens2mask(lens), n_best)
        threshold = n_best if intents.size(-1) > 2 * n_best else int(n_best / 2)
        intent_scores, intent_idxs = intents.topk(threshold, dim=1)

        comb_scores = slot_scores.unsqueeze(-1) + intent_scores.unsqueeze(1)
        flat_comb_scores = comb_scores.contiguous().view(comb_scores.size(0), -1)
        _, best_score_id = flat_comb_scores.topk(n_best, 1, True, True)
        pick_slots = best_score_id // intent_scores.size(-1) # bsize x n_best
        pick_intents = best_score_id - pick_slots * intent_scores.size(-1)

        slot_scores = torch.gather(slot_scores, 1, pick_slots)
        slot_idxs = [
            (torch.gather(b, 0, pick_slots[idx].unsqueeze(dim=1).repeat(1, b.size(1)))).tolist()
            for idx, b in enumerate(slot_idxs)
        ]
        intent_scores = torch.gather(intent_scores, 1, pick_intents)
        intent_idxs = torch.gather(intent_idxs, 1, pick_intents).tolist()
        return {"intent": (intent_scores, intent_idxs), "slot": (slot_scores, slot_idxs)}
