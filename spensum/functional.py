import torch
from torch.autograd import Variable

def sequence_dropout(input, p=.5, training=False, batch_first=False):

    if training and p > 0:
        if batch_first:
            dims = input[:,:1].size()
            seq_size = input.size(1)
            mask = Variable(
                torch.bernoulli(input.data.new(dims).fill_(1 - p)).repeat(
                    1, seq_size, 1))
        else:
            dims = input[:1].size()
            seq_size = input.size(0)
            mask = Variable(
                torch.bernoulli(input.data.new(dims).fill_(1 - p)).repeat(
                    seq_size, 1, 1))

        masked_input = input * mask
        masked_input_rescaled = masked_input.div_(p)
        return masked_input_rescaled

    else:
        return input
