# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-12-04 23:19:38
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-12-16 22:15:56
# https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M

class CRF(nn.Module):

    def __init__(self, tagset_size, start_idx=-2, end_idx=-1):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size if start_idx != -2 else tagset_size + 2
        # We add 2 if vocab does not contain <bos> and <eos>
        self.start_idx, self.end_idx = start_idx, end_idx
        # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size, self.tagset_size)
        init_transitions[:, self.start_idx] = -10000.0
        init_transitions[self.end_idx, :] = -10000.0
        # init_transitions[:, pad_idx] = -10000.0  ### 0 is <pad>
        # init_transitions[pad_idx, :] = -10000.0  ### 0 is <pad>
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *from* i *to* j.
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_alg(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size
        mask = mask.transpose(1, 0).contiguous()

        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)
        ## scores: x -> y_t  plus y_{t-1} -> y_t
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_idx, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        for idx, cur_values in seq_iter:
            # cur_values: bat_size * from_target * to_target
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = torch.logsumexp(cur_values, 1) # bat_size * to_target

            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in self.end_idx
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = torch.logsumexp(cur_values, 1)
        final_partition = cur_partition[:, self.end_idx]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        device = feats.device
        assert tag_size == self.tagset_size
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask, dim = 1).view(batch_size,1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = ~ mask
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_idx, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider self.start_idx/self.end_idx
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp = cur_bp * (~ mask[idx].view(batch_size, 1).expand(batch_size, tag_size)).long()
            # cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        ### add score to final self.end_idx
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,0).contiguous() ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        ### calculate the score from last partition to end state (and then select the self.end_idx from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        last_score, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size).to(device, dtype=torch.long)
        back_points.append(pad_zero)
        back_points  =  torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in self.end_idx
        pointer = last_bp[:, self.end_idx]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1,0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = torch.zeros(seq_len, batch_size, dtype=torch.long, device=device)
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1)).squeeze(1)
            decode_idx[idx] = pointer.detach().view(batch_size)
        decode_idx = decode_idx.transpose(1,0)
        ## calculate path score
        score = last_score[:, self.end_idx]
        all_scores = last_values[:, :, self.end_idx]
        max_score, _ = torch.max(all_scores, 1)
        minus_scores = all_scores - max_score.unsqueeze(-1)
        logsumexp_score = torch.logsumexp(minus_scores, dim=1)
        path_score = score - max_score - logsumexp_score
        return path_score.unsqueeze(1), decode_idx.unsqueeze(1)

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        device = scores.device
        ## convert tag value into a new format, recorded label bigram information to index
        new_tags = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for idx in range(seq_len):
            if idx == 0:
                ## from self.start_idx to first label, use (tag_size - 2) since self.start_idx = -2
                new_tags[:,0] =  (tag_size - 2)*tag_size + tags[:,0]

            else:
                new_tags[:,idx] =  tags[:,idx-1]*tag_size + tags[:,idx]
        new_tags = new_tags.to(device)

        ## transition for label to self.end_idx
        end_transition = self.transitions[:,self.end_idx].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask, dim = 1).view(batch_size,1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to self.end_idx
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1,0))

        ## calculate the score from self.start_idx to first label
        # #start_transition = self.transitions[self.start_idx,:].view(1, tag_size).expand(batch_size, tag_size)
        # #start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # #gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        # batch_size = feats.size(0)
        forward_score, scores = self._calculate_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        device = feats.device
        assert tag_size == self.tagset_size
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask, dim = 1).view(batch_size,1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = ~ mask
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_idx, :].clone()  # bat_size * to_target_size
        ## initial partition [batch_size, tag_size]
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                ## compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size*nbest, tag_size)
                # print "cur size:",cur_values.size()
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            ## cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
            # print partition[:,0,:]
            # print cur_bp[:,0,:]
            # print "nbest, ",idx
            if idx == 1:
                cur_bp = cur_bp*nbest
            partition = partition.transpose(2,1)
            cur_bp = cur_bp.transpose(2,1)

            # print partition
            # exit(0)
            #partition: (batch_size * to_target * nbest)
            #cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            ## cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            ## mask[idx] ? mask[idx-1]
            cur_bp = cur_bp * (~ mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest)).long()
            # cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
            # print cur_bp[0]
            back_points.append(cur_bp)
        ### add score to final self.end_idx
        partition_history = torch.cat(partition_history,0).view(seq_len, batch_size, tag_size, nbest).transpose(1,0).contiguous() ## (batch_size, seq_len, nbest, tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size,1,1,1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        ### calculate the score from last partition to end state (and then select the self.end_idx from it)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size*nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        ## end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2,1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = torch.zeros(batch_size, tag_size, nbest).to(device, dtype=torch.long)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        ## select end ids in self.end_idx
        pointer = end_bp[:, self.end_idx, :] ## (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1,0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last[0]
        # exit(0)
        ## copy the ids of last position:insert_last to back_points, though the last_position index
        ## last_position includes the length of batch sentences
        # print "old:", back_points[9,0,:,:]
        back_points.scatter_(1, last_position, insert_last)
        ## back_points: [batch_size, seq_length, tag_size, nbest]
        # print "new:", back_points[9,0,:,:]
        # exit(0)
        # print pointer[2]
        '''
        back_points: in simple demonstratration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        '''

        back_points = back_points.transpose(1,0).contiguous()
        # print back_points[0]
        ## back_points: (seq_len, batch, tag_size, nbest)
        ## decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = torch.zeros(seq_len, batch_size, nbest, dtype=torch.long, device=device)
        decode_idx[-1] = pointer.data // nbest
        # print "pointer-1:",pointer[2]
        # exit(0)
        # use old mask, let 0 means has token
        for idx in range(len(back_points)-2, -1, -1):
            # print "pointer: ",idx,  pointer[3]
            # print "back:",back_points[idx][3]
            # print "mask:",mask[idx+1,3]
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size*nbest), 1, pointer.contiguous().view(batch_size,nbest))
            decode_idx[idx] = new_pointer.data // nbest
            # # use new pointer to remember the last end nbest ids for non longest
            pointer = new_pointer + pointer.contiguous().view(batch_size,nbest)*mask[idx].view(batch_size,1).expand(batch_size, nbest).long()

        ## decode_idx: [batch, seq_len, nbest]
        decode_idx = decode_idx.transpose(1,0).transpose(1, 2)
        # print decode_idx[:,:,0]
        # print "nbest:",nbest
        # print "diff:", decode_idx[:,:,0]- decode_idx[:,:,4]
        # print decode_idx[:,0,:]
        # exit(0)

        ### calculate probability for each sequence
        scores = end_partition[:, :, self.end_idx]
        ## scores: [batch_size, nbest]
        max_scores,_ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size,1).expand(batch_size, nbest)
        path_score = F.log_softmax(minus_scores, 1)
        ## path_score: [batch_size, nbest]
        # exit(0)
        return path_score, decode_idx
