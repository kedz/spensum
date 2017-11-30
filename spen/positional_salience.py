import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalSalience(nn.Module):
    def __init__(self, input_size, num_positions, mask_value=-1):
        super(PositionalSalience, self).__init__()
        self.weight = nn.Parameter(torch.rand(num_positions, input_size))
        self.bias = nn.Parameter(torch.ones(num_positions))
        self.mask_value_ = mask_value
        self.num_positions_ = num_positions
        self.input_size_ = input_size

    @property
    def num_positions(self):
        return self.num_positions_

    @property
    def input_size(self):
        return self.input_size_

    @property
    def mask_value(self):
        return self.mask_value_

    def forward(self, inputs):
        mask = inputs.data[:,:,0].eq(self.mask_value)
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        if self.weight.size(0) < input_size:
            leftover = input_size - self.weight.size(0)
            weight = torch.cat(
                [self.weight] + [self.weight[-1:].repeat(leftover, 1)], 0)
            bias = torch.cat([self.bias] + [self.bias[-1].repeat(leftover)])
        elif input_size < self.weight.size(0):
            weight = self.weight[:input_size]
            bias = self.bias[:input_size]
        else:
            weight = self.weight
            bias = self.bias

        batch_weight = weight.view(1, input_size, self.input_size).repeat(
            batch_size, 1, 1)
        batch_bias = bias.view(1, input_size).repeat(batch_size, 1)

        logits_flat = torch.bmm(
            inputs.view(batch_size * input_size, 1, self.input_size),
            batch_weight.view(batch_size * input_size, self.input_size, 1))
        
        logits = logits_flat.view(batch_size, input_size) + batch_bias
        prob = F.sigmoid(logits)
        prob.data.masked_fill_(mask, 0)
        return prob
