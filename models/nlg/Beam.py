from __future__ import division
import torch
from models.nlg import penalties
from utils.constants import *

class Beam(object):
    """
        Class for managing the internals of the beam search process.
        Takes care of beams, back pointers, and scores. (Revised from OpenNMT.)
        @args:
            size (int): beam size
            vocab (dict): obtain indices of padding, beginning, and ending.
            global_scorer (:obj:`GlobalScorer`)
            exclusion_tokens (torch.FloatTensor): input slots one-hot index
            device (torch.device)
    """
    def __init__(self, size, vocab, device=None,
                 global_scorer=None, min_length=0,
                 top_k=0, penalty=0.0, exclusion_tokens=None):

        self.size = size
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.zeros(size, dtype=torch.float, device=self.device)

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.zeros(size, dtype=torch.long, device=self.device).fill_(vocab[PAD])]
        self.next_ys[0][0] = vocab[BOS]

        # Has EOS topped the beam yet.
        self._eos = vocab[EOS]
        self.eos_top = False

        # Other special symbols
        self._bos = vocab[BOS]
        self._pad = vocab[PAD]
        self._equal = vocab[EQUAL]

        # Time and k pair for finished.
        self.finished = []

        # Information for global scoring.
        self.global_scorer = global_scorer

        # Minimum prediction length, 3: at least one symbol except BOS and EOS
        self.min_length = max([min_length + 1, 2])

        if exclusion_tokens is not None:
            extra_ones = torch.ones(len(vocab) - exclusion_tokens.size(0), dtype=torch.float).to(self.device)
            exclusion_tokens = torch.cat([exclusion_tokens, extra_ones], dim=0)
            self.exclusion_tokens = exclusion_tokens == 0.
        else:
            self.exclusion_tokens = None
        
        self.top_k = int(top_k) if top_k >= 2 and top_k <= self.size else self.size
        assert penalty >= 0. and penalty <= 1.0
        self.penalty = - penalty * torch.arange(self.size, dtype=torch.float, device=self.device)

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs):
        """
        Given prob over words for every last beam `wordLk`

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        masks = torch.zeros(word_probs.size(), requires_grad=False, dtype=torch.float, device=self.device)
        masks[:, self._bos] = 1e20
        masks[:, self._pad] = 1e20
        masks[:, self._equal] = 1e20
        if self.exclusion_tokens is not None:
            masks.masked_fill_(self.exclusion_tokens, 1e20)
        if cur_len < self.min_length:
            masks[:, self._eos] = 1e20
        word_probs = word_probs - masks

        # pick top_k candidates
        cur_top_k = self.size if len(self.prev_ks) == 0 else self.top_k
        top_k, sort_key = word_probs.topk(cur_top_k, -1, True, True)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = top_k + self.scores.unsqueeze(1)
            masks = torch.zeros(beam_scores.size(), requires_grad=False, dtype=torch.float, device=self.device)
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    masks[i] = 1e20
            beam_scores = beam_scores - masks
        else:
            beam_scores = top_k[0]
        rank_beam_scores = beam_scores + self.penalty[:cur_top_k]
        flat_beam_scores = rank_beam_scores.contiguous().view(-1)
        _, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // cur_top_k
        self.prev_ks.append(prev_k)
        next_y = torch.take(sort_key.contiguous().view(-1), best_scores_id)
        self.next_ys.append(next_y)
        self.scores = torch.take(beam_scores.contiguous().view(-1), best_scores_id)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores) # normalize score by length penalty
                rank_s, s = global_scores[i], self.scores[i]
                self.finished.append(([rank_s, s], len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.eos_top = True
        
        return self.done()

    def done(self):
        return self.eos_top and len(self.finished) >= self.size

    def sort_best(self):
        """
            Sort the current beam.
        """
        return torch.sort(self.scores, 0, True) # beam size
    
    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                rank_s, s = global_scores[i], self.scores[i]
                self.finished.append(([rank_s, s], len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0][0])
        scores = [sc[1] for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_temporary_hyp(self, k):
        """
            Get current hypotheses of rank k ( 0 <= rank <= beam_size-1 ). 
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return torch.stack(hyp[::-1])

    def get_hyp(self, timestep, k):
        """ 
            Walk back to construct the full hypothesis. 
            hyp contains </s> but does not contain <s>
            @return:
                hyp: LongTensor of size tgt_len
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return torch.stack(hyp[::-1])

class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, len_penalty):
        self.alpha = alpha
        penalty_builder = penalties.PenaltyBuilder(len_penalty)
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam, logprobs, self.alpha)
        return normalized_probs
