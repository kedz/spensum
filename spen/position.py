import torch
import torch.nn as nn
import torch.nn.functional as F


class Position(nn.Module):
    def __init__(self, num_positions):
        super(Position, self).__init__()
        self.weight = nn.Parameter(torch.rand(num_positions))
        self.num_positions_ = num_positions
    
    @property
    def num_positions(self):
        return self.num_positions_

    def forward(self, inputs):
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        if self.weight.size(0) < input_size:
            leftover = input_size - self.weight.size(0)
            weight = torch.cat(
                [self.weight] + [self.weight[-1].repeat(leftover)])
        elif input_size < self.weight.size(0):
            weight = self.weight[:input_size]
        else:
            weight = self.weight
        batch_weight = weight.view(1, input_size).repeat(batch_size, 1)
        return F.sigmoid(batch_weight)
