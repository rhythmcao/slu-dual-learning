#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def tile(x, count, dim=0):
    """
        Tiles x on dimension dim count times.
        E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
            [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
        Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

def lens2mask(lens):
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks

def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, [bsize x max_seq_len x in_dim]
            lens(torch.LongTensor): seq len for each sample, allow length=0, padding with 0-vector, [bsize]
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_dim*2
            hidden_states([tuple of ]torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_dim
    """
    # rerank according to lens and remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_num, total_num = torch.sum(sorted_lens > 0).item(), sorted_lens.size(0)
    sort_key = sort_key[:nonzero_num]
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key)
    # forward non empty inputs    
    packed_inputs = rnn_utils.pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_num].tolist(), batch_first=True)
    packed_out, sorted_h = encoder(packed_inputs)  # bsize x srclen x dim
    sorted_out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        sorted_h, sorted_c = sorted_h
    # rerank according to sort_key
    out_shape = list(sorted_out.size())
    out_shape[0] = total_num
    out = sorted_out.new_zeros(*out_shape).scatter_(0, sort_key.unsqueeze(-1).unsqueeze(-1).repeat(1, *out_shape[1:]), sorted_out)
    h_shape = list(sorted_h.size())
    h_shape[1] = total_num
    h = sorted_h.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_h)
    if cell.upper() == 'LSTM':
        c = sorted_c.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_c)
        return out, (h.contiguous(), c.contiguous())
    return out, h.contiguous()

class PoolingFunction(nn.Module):
    """ Map a sequence of hidden_size dim vectors into one fixed size vector with dimension output_size """
    def __init__(self, hidden_size=256, output_size=256, bias=True, method='attentive-pooling'):
        super(PoolingFunction, self).__init__()
        assert method in ['mean-pooling', 'max-pooling', 'attentive-pooling']
        self.method = method
        if self.method == 'attentive-pooling':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=bias)
            )
        self.mapping_function = nn.Sequential(nn.Linear(hidden_size, output_size, bias=bias), nn.Tanh()) \
            if hidden_size != output_size else lambda x: x

    def forward(self, inputs, mask=None):
        """ @args:
                inputs(torch.FloatTensor): features, batch_size x seq_len x hidden_size
                mask(torch.BoolTensor): mask for inputs, batch_size x seq_len
            @return:
                outputs(torch.FloatTensor): aggregate seq_len dim for inputs, batch_size x output_size
        """
        if self.method == 'max-pooling':
            outputs = inputs.masked_fill(~ mask.unsqueeze(-1), -1e8)
            outputs = outputs.max(dim=1)[0]
        elif self.method == 'mean-pooling':
            mask_float = mask.float().unsqueeze(-1)
            outputs = (inputs * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        elif self.method == 'attentive-pooling':
            e = self.attn(inputs).squeeze(-1)
            e = e + (1 - mask.float()) * (-1e20)
            a = torch.softmax(e, dim=1).unsqueeze(1)
            outputs = torch.bmm(a, inputs).squeeze(1)
        else:
            raise ValueError('[Error]: Unrecognized pooling method %s !' % (self.method))
        outputs = self.mapping_function(outputs)
        return outputs