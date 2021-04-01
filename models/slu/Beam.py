from __future__ import division
import torch
from utils.constants import PAD, BOS

class Beam(object):
    """
    Takes care of beams, back pointers, and scores. (Revised from OpenNMT)
    Add top-k capping and inner order in one step.
    @args:
        size (int): beam size
        vocab (dict): contains symbol BOS, PAD
        length (int): target sequence length
        top_k (int): only top k candidates in each step, [2, beam size],
            except first step (k = beam size), by default beam search size
        penalty (float): consider inner order in a beam, [0, 1],  by default penalty is 0
        device (torch.device)
    """

    def __init__(self, size, vocab, length, top_k=0, penalty=0., device=None):
        assert length > 0 and size > 2
        self.size = size
        self.length = length
        self.top_k = int(top_k) if top_k >= 2 and top_k <= self.size else self.size
        assert penalty >= 0. and penalty <= 1.0
        self.penalty = - penalty * torch.arange(self.size, dtype=torch.float, device=device)
        self.device = device
        # The score for each translation on the beam
        self.scores = torch.zeros(size, dtype=torch.float, device=self.device)

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.zeros(size, dtype=torch.long, device=self.device).fill_(vocab[PAD])]
        self.next_ys[0][0] = vocab[BOS]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs):
        """
        Given prob over words for every last beam

        Parameters:

        * `word_probs`- probs of advancing from the last step ([K x] slot num)

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(-1)
        cur_top_k = self.size if len(self.prev_ks) == 0 else self.top_k
        top_k, sort_key = word_probs.topk(cur_top_k, -1, True, True)
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = top_k + self.scores.unsqueeze(1) # broadcast mechanism
        else:
            beam_scores = top_k if word_probs.dim() == 1 else top_k[0]
        rank_beam_scores = beam_scores + self.penalty[:cur_top_k]
        flat_beam_scores = rank_beam_scores.contiguous().view(-1)
        _, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

        # best_scores_id is flattened beam x cur_top_k array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // cur_top_k
        self.prev_ks.append(prev_k)
        if sort_key.dim() == 1:
            sort_key = sort_key.unsqueeze(0).repeat(self.size, 1)
        next_y = torch.take(sort_key.contiguous().view(-1), best_scores_id)
        self.next_ys.append(next_y)
        self.scores = torch.take(beam_scores.contiguous().view(-1), best_scores_id)
        return self.done()

    def done(self):
        return len(self.prev_ks) == self.length

    def sort_best(self):
        """
            Sort the current beam.
        """
        return torch.sort(self.scores, 0, True)[0] # beam size

    def get_hyp(self, k):
        """
            Get current hypotheses of rank k ( 0 <= rank <= beam_size-1 ).
        """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return torch.stack(hyp[::-1])
