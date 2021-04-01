#coding=utf8
import torch
import torch.nn as nn

def set_celoss_function(ignore_index=-100):
    loss_function = ConfidenceAwareNLLLoss(ignore_index=ignore_index)
    return loss_function

class ConfidenceAwareNLLLoss(nn.Module):

    def __init__(self, *args, **kargs):
        super(ConfidenceAwareNLLLoss, self).__init__()
        kargs['reduction'] = 'none'
        self.loss_function = nn.NLLLoss(*args, **kargs)

    def forward(self, inputs, targets, conf=None):
        if conf is None: # confidence for each training sample
            conf = torch.ones(inputs.size(0), dtype=torch.float).to(inputs.device)
        if inputs.dim() == 2: # intent detection
            loss = self.loss_function(inputs, targets)
        else: # slot filling
            assert inputs.dim() == 3
            bsize, seq_len, voc_size = list(inputs.size())
            loss = self.loss_function(inputs.contiguous().view(-1, voc_size), targets.contiguous().view(-1))
            loss = loss.contiguous().view(bsize, seq_len).sum(dim=1)
        return (loss * conf).sum()

def set_scloss_function(slot_weight=1.0, eta=1e-4, xi=100):
    return SlotControlLoss(slot_weight, eta, xi)

class SlotControlLoss(nn.Module):

    def __init__(self, slot_weight=1.0, eta=1e-4, xi=100):
        super(SlotControlLoss, self).__init__()
        self.slot_weight = slot_weight
        self.eta = eta
        self.xi = xi

    def forward(self, slot_hist, out_lens, conf=None):
        """
        @args:
            slot_hist(torch.FloatTensor): including s0, bsize x seq_len x slot_size
            out_lens(torch.LongTensor): including BOS/EOS, bsize
        @return:
            sc_loss
        """
        if slot_hist is None:
            return torch.tensor(0, dtype=torch.float).to(out_lens.device)
        if conf is None:
            conf = slot_hist.new_ones(slot_hist.size(0))
        out_lens = out_lens - 1
        sT = torch.gather(slot_hist, dim=1, index=out_lens.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, slot_hist.size(-1))).squeeze(1)
        sT_norm = torch.norm(sT, p=2, dim=1) # bsize
        diff = slot_hist[:, 1:, :] - slot_hist[:, :-1, :]
        diff = [
            torch.pow(slot_hist.new_full((out_lens[idx].item(), ), self.xi),
                torch.norm(ex[:out_lens[idx].item()], p=2, dim=1)).sum()
            for idx, ex in enumerate(diff)
        ]
        diff = torch.stack(diff, dim=0)
        return self.slot_weight * (torch.sum(sT_norm * conf) + self.eta * torch.sum(diff * conf))