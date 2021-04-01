#coding=utf8
import torch
import torch.nn as nn
from models.nlg.Beam import Beam, GNMTGlobalScorer
from utils.constants import *
from models.model_utils import tile

class NLGModel(nn.Module):

    def __init__(self, intent_embed, word_embed, encoder, decoder, enc2dec, generator):
        super(NLGModel, self).__init__()
        self.intent_embed = intent_embed
        self.word_embed = word_embed
        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.generator = generator

    def forward(self, *args, **kargs):
        raise NotImplementedError

    def pad_embedding_grad_zero(self):
        self.word_embed.pad_embedding_grad_zero()

    def decode_batch(self, intents, slots, slot_lens, lens, slot_states, copy_tokens, vocab, beam=5, n_best=1, **kargs):
        intents = self.intent_embed(intents)
        memory = self.encoder(self.word_embed(slots), slot_lens, lens)
        init_states = self.enc2dec(memory, slot_lens, intents)
        init_states = (init_states[0].contiguous().transpose(0, 1), init_states[1].contiguous().transpose(0, 1))
        if beam == 1:
            return self.decode_greed(init_states, memory, slot_lens, intents, slot_states, copy_tokens, vocab)
        else:
            return self.decode_beam_search(init_states, memory, slot_lens, intents, slot_states, copy_tokens, vocab, beam, n_best, **kargs)

    def decode_one_step(self, ys, intents, memory, dec_states, slot_states, slot_lens, copy_tokens):
        raise NotImplementedError

    def decode_greed(self, init_states, memory, slot_lens, intents, slot_states, copy_tokens, vocab):
        """
        @args:
            init_states: decoder initial hidden states, bsize x num_layers x hidden_dim
            memory: slot encoder representation, bsize x max_slot_num x slot_dim
            slot_lens: number of slots for each batch sample, bsize
            intents: intents representations, bsize x intent_dim
            slot_states: slot one-hot vector, bsize x slot_size
            copy_tokens: mapping each slot into target vocabulary, bsize x max_slot_num x vocab_size
            vocab: target vocabulary
        @return:
            results(dict):
                predictions: idx list, e.g. [ [[0, 3, 4, 5]] , [[0, 2, 4, 5]] , ... ]
                scores: logscores tensor, bsize x n_best, n_best=1
        """
        results = {"predictions": [], "scores": 0}
        ys = slot_lens.new_ones(slot_lens.size(0), 1).fill_(vocab[BOS])
        dec_states = init_states
        # record whether each batch sample is finished
        all_done = torch.tensor([False] * slot_lens.size(0), dtype=torch.bool, device=slot_lens.device)
        scores = torch.zeros(slot_lens.size(0), 1, dtype=torch.float, device=slot_lens.device)
        predictions = [[] for i in range(slot_lens.size(0))]
        for i in range(DECODE_MAX_LENGTH):
            logprob, (dec_states, slot_states) = self.decode_one_step(
                ys, intents, memory, dec_states, slot_states, slot_lens, copy_tokens)
            maxprob, ys = torch.max(logprob, dim=1, keepdim=True)
            for i in range(slot_lens.size(0)):
                if not all_done[i]:
                    scores[i] += maxprob[i]
                    predictions[i].append(ys[i])
            done = ys.squeeze(dim=1) == vocab[EOS]
            all_done |= done
            if all_done.all():
                break
        results["predictions"], results["scores"] = [[torch.cat(pred).tolist()] for pred in predictions], scores
        return results

    def decode_beam_search(self, init_states, memory, slot_lens, intents, slot_states, copy_tokens, vocab, beam=5, n_best=1, length_pen='avg', top_k=0, penalty=0.):
        results = {"scores":[], "predictions":[]}
        remaining_sents = slot_lens.size(0)
        global_scorer = GNMTGlobalScorer(0.6, length_pen)
        # Construct beams
        beams = [ Beam(beam, vocab, device=slot_lens.device,
                        global_scorer = global_scorer,
                        min_length = slot_lens[i].item(),
                        exclusion_tokens = slot_states[i],
                        top_k=top_k, penalty=penalty)
                for i in range(remaining_sents) ]

        # repeat beam times
        memory, slot_lens, intents, hidden_states, slot_states, copy_tokens = tile([memory, slot_lens, intents, init_states, slot_states, copy_tokens], beam, dim=0)
        batch_idx = list(range(remaining_sents))

        for i in range(DECODE_MAX_LENGTH):
            # (a) construct beamsize * remaining_sents next words
            ys = torch.stack([b.get_current_state() for b in beams if not b.done()]).contiguous().view(-1,1)

            # (b) pass through the decoder network
            logprob, (hidden_states, slot_states) = self.decode_one_step(ys, intents, memory, hidden_states, slot_states, slot_lens, copy_tokens)
            out = logprob.contiguous().view(remaining_sents, beam, -1)

            # (c) advance each beam
            active, select_indices_array = [], []
            # Loop over the remaining_batch number of beam
            for b in range(remaining_sents):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beams[idx].advance(out[b])
                if not beams[idx].done():
                    active.append((idx, b))
                select_indices_array.append(beams[idx].get_current_origin() + b * beam)

            # (d) update hidden_states and slot_states history
            select_indices_array = torch.cat(select_indices_array, dim=0)
            hidden_states = (hidden_states[0].index_select(0, select_indices_array), hidden_states[1].index_select(0, select_indices_array))
            slot_states = slot_states.index_select(0, select_indices_array)

            if not active:
                break

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=slot_lens.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            def update_active(t):
                if t is None:
                    return t
                elif type(t) in [list, tuple]:
                    return type(t)([update_active(each) for each in t])
                else:
                    t_reshape = t.contiguous().view(remaining_sents, beam, -1)
                    new_size = list(t.size())
                    new_size[0] = -1
                    return t_reshape.index_select(0, active_idx).view(*new_size)

            memory, slot_lens, intents, hidden_states, slot_states, copy_tokens = \
                update_active([memory, slot_lens, intents, hidden_states, slot_states, copy_tokens])
            remaining_sents = len(active)

        for b in beams:
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp.tolist()) # hyp contains </s> but does not contain <s>
            results["predictions"].append(hyps) # list of variable_tgt_len
            results["scores"].append(torch.stack(scores)[:n_best])
        results["scores"] = torch.stack(results["scores"])
        return results