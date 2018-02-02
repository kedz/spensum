import torch
import torch.nn as nn
from torch.autograd import Variable


class SequenceStandardizer(nn.Module):
    def __init__(self, input_size, dim=1):
        super(SequenceStandardizer, self).__init__()
        self.dim = dim
        self.output_size = input_size

    def forward(self, inputs):
        
        mean = inputs.sequence.data.new(
            inputs.sequence.size(0), 1, self.output_size)
        sample_std = mean.new(mean.size())
        
        for batch in range(inputs.sequence.size(0)):
            length = inputs.length.data[batch]
            mean[batch,0].copy_(inputs.sequence.data[batch,:length].sum(0))
            mean[batch,0].div_(length)
            sq = (inputs.sequence.data[batch,:length] - mean[batch])**2
            sample_std[batch,0].copy_(sq.sum(0))
            sample_std[batch,0].div_(length - 1)
            sample_std[batch,0].sqrt_()

        mean = Variable(mean) 
        sample_std = Variable(sample_std)
        return (inputs.sequence - mean) / sample_std
