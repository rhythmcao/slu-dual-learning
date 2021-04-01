#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slu.Beam import Beam
from models.model_utils import tile
from utils.constants import BOS

class FocusModel(nn.Module):

    def __init__(self, word_embed, encoder, enc2dec, slot_embed, slot_decoder, intent_decoder):
        super(FocusModel, self).__init__()
        self.word_embed = word_embed
        self.slot_embed = slot_embed
        self.encoder = encoder
        self.enc2dec = enc2dec
        self.slot_decoder = slot_decoder
        self.intent_decoder = intent_decoder

    def pad_embedding_grad_zero(self):
        self.word_embed.pad_embedding_grad_zero()
        self.slot_embed.pad_embedding_grad_zero()

    def forward(self, inputs, lens, dec_inputs=None):
        outputs, hidden_states = self.encoder(self.word_embed(inputs), lens)
        intents = self.intent_decoder(hidden_states, outputs, lens)
        hidden_states = self.enc2dec(hidden_states)
        dec_inputs = self.slot_embed(dec_inputs[:, :-1]) # teacher forcing
        bios, _ = self.slot_decoder(dec_inputs, outputs, hidden_states)
        return bios, intents

    def decode_batch(self, inputs, lens, vocab, beam=5, n_best=1, **kargs):
        outputs, hidden_states = self.encoder(self.word_embed(inputs), lens)
        intents = self.intent_decoder(hidden_states, outputs, lens)
        hidden_states = self.enc2dec(hidden_states)
        if beam == 1:
            return self.decode_greed(intents, outputs, hidden_states, lens, vocab)
        else:
            return self.decode_beam_search(intents, outputs, hidden_states, lens, vocab, beam, n_best, **kargs)

    def decode_greed(self, intents, memory, hidden_states, lens, vocab):
        """
        @args:
            intents(torch.FloatTensor): bsize x intent_num
            memory(torch.FloatTensor): encoder output, bsize x seqlen x hidden_size
            hidden_states([tuple of] torch.FloatTensor): num_layer x bsize x hidden_size
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
        ys = torch.ones(memory.size(0), 1).fill_(vocab[BOS]).long().to(memory.device)
        all_done = torch.tensor([False] * memory.size(0), dtype=torch.uint8, device=memory.device)
        scores = torch.zeros(memory.size(0), 1, dtype=torch.float, device=memory.device)
        predictions = [[] for i in range(memory.size(0))]
        for i in range(max(lens.tolist())):
            inputs = self.slot_embed(ys)
            logprob, hidden_states = self.slot_decoder(inputs, memory[:, i:i+1, :], hidden_states)
            logprob = logprob.squeeze(dim=1)
            maxprob, ys = torch.max(logprob, dim=1, keepdim=True)
            for i in range(memory.size(0)):
                if not all_done[i]:
                    scores[i] += maxprob[i]
                    predictions[i].append(ys[i])
            done = lens == (i + 1)
            all_done |= done
        predictions = [[torch.cat(pred).tolist()] for pred in predictions]
        int_score, int_idx = torch.max(intents, dim=1, keepdim=True)
        return {"intent": (int_score, int_idx.tolist()), "slot": (scores, predictions)}

    def decode_beam_search(self, intents, memory, hidden_states, lens, vocab, beam_size=5, n_best=1, top_k=0, penalty=0.):
        """
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
        remaining_sents = lens.size(0)
        beams = [Beam(beam_size, vocab, l, top_k=top_k, penalty=penalty, device=intents.device) for l in lens_list]
        memory = tile(memory, beam_size, dim=0)
        hidden_states = tile(hidden_states, beam_size, dim=1)
        h_c = type(hidden_states) in [list, tuple]
        batch_idx = list(range(remaining_sents))
        for i in range(max(lens_list)):
            # (a) construct beamsize * remaining_sents next words
            ys = torch.stack([b.get_current_state() for b in beams if not b.done()]).contiguous().view(-1,1)

            # (b) pass through the decoder network
            inputs = self.slot_embed(ys)
            logprob, hidden_states = self.slot_decoder(inputs, memory[:, i:i+1, :], hidden_states)
            out = logprob.contiguous().view(remaining_sents, beam_size, -1)

            # (c) advance each beam
            active, select_indices_array = [], []
            # Loop over the remaining_batch number of beam
            for b in range(remaining_sents):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beams[idx].advance(out[b])
                if not beams[idx].done():
                    active.append((idx, b))
                select_indices_array.append(beams[idx].get_current_origin() + b * beam_size)

            # (d) update hidden_states history
            select_indices_array = torch.cat(select_indices_array, dim=0)
            if h_c:
                hidden_states = (hidden_states[0].index_select(1, select_indices_array), hidden_states[1].index_select(1, select_indices_array))
            else:
                hidden_states = hidden_states.index_select(1, select_indices_array)

            if not active:
                break

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=memory.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            def update_active(t):
                if t is None:
                    return t
                elif type(t) in [list, tuple]:
                    return type(t)([update_active(each) for each in t])
                else:
                    t_reshape = t.contiguous().view(remaining_sents, beam_size, -1)
                    new_size = list(t.size())
                    new_size[0] = -1
                    return t_reshape.index_select(0, active_idx).view(*new_size)

            if h_c:
                hidden_states = (
                    update_active(hidden_states[0].transpose(0, 1)).transpose(0, 1).contiguous(),
                    update_active(hidden_states[1].transpose(0, 1)).transpose(0, 1).contiguous()
                )
            else:
                hidden_states = update_active(hidden_states.transpose(0, 1)).transpose(0, 1).contiguous()
            memory = update_active(memory)
            remaining_sents = len(active)

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
