#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slu.Beam import Beam

class BiRNNModel(nn.Module):

    def __init__(self, embed, encoder, slot_decoder, intent_decoder):
        super(BiRNNModel, self).__init__()
        self.word_embed = embed
        self.encoder = encoder
        self.slot_decoder = slot_decoder
        self.intent_decoder = intent_decoder

    def pad_embedding_grad_zero(self):
        self.word_embed.pad_embedding_grad_zero()

    def forward(self, inputs, lens, dec_inputs=None):
        outputs, hidden_states = self.encoder(self.word_embed(inputs), lens)
        bios = self.slot_decoder(outputs)
        intents = self.intent_decoder(hidden_states, outputs, lens)
        return bios, intents

    def decode_batch(self, inputs, lens, vocab, beam=5, n_best=1, **kargs):
        bios, intents = self.forward(inputs, lens)
        if beam == 1:
            return self.decode_greed(bios, intents, lens)
        else:
            return self.decode_beam_search(bios, intents, lens, vocab, beam, n_best, **kargs)

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
        bios_score, bios_idx = torch.max(bios, dim=-1)
        slot_idx, slot_score, lens_list = [], [], lens.tolist()
        for idx in range(bios_idx.size(0)):
            l = lens_list[idx]
            slot_idx.append([bios_idx[idx, :l].tolist()])
            slot_score.append(torch.sum(bios_score[idx, :l]))
        slot_score = torch.stack(slot_score).unsqueeze(dim=1)
        intent_score, intent_idx = torch.max(intents, dim=1, keepdim=True)
        return {"intent": (intent_score, intent_idx.tolist()), "slot": (slot_score, slot_idx)}

    def decode_beam_search(self, bios, intents, lens, vocab, beam_size=5, n_best=1, top_k=0, penalty=0.):
        """
        @args:
            vocab(dict): map output word to id
        @return:
            dict:
            (key)intent: (value) tuple of (n_best most likely intent score, n_best most likely intent idx)
                intent_score(torch.FloatTensor): bsize x n_best
                intent_idx(list): bsize x n_best
            (key)slot: (value) tuple of (n_best most likely seq score, n_best most likely seq idx)
                slot_score(torch.FloatTensor): bsize x n_best
                slot_idx(list): n_best=2, such as [ [[1, 2, 4, 9], [1, 2, 3, 5]] , [[1, 2, 5], [1, 2, 3]] , ... ]
        """
        lens_list = lens.tolist()
        beams = [Beam(beam_size, vocab, l, top_k=top_k, penalty=penalty, device=bios.device) for l in lens_list]
        for t in range(bios.size(1)):
            for idx, b in enumerate(beams):
                if not b.done():
                    b.advance(bios[idx, t, :])
        slot_scores = torch.stack([b.sort_best() for b in beams]) # bsize x beam_size
        slot_idxs = [torch.stack([b.get_hyp(i) for i in range(beam_size)]) for b in beams]
        threshold = beam_size if intents.size(-1) > 2 * beam_size else int(beam_size / 2)
        intent_scores, intent_idxs = intents.topk(threshold, dim=1)

        comb_scores = slot_scores.unsqueeze(-1) + intent_scores.unsqueeze(1)
        flat_comb_scores = comb_scores.contiguous().view(len(beams), -1)
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
