from .spen_module import SpenModule
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Position(SpenModule):
    def __init__(self, num_positions=50, name="Position", mask_value=-1,
                 burn_in=0):
        super(Position, self).__init__(
            name=name, mask_value=mask_value, burn_in=burn_in)
        self.num_positions_ = num_positions
        self.embedding = nn.Embedding(num_positions + 1, 1, padding_idx=0)
    
    @property
    def num_positions(self):
        return self.num_positions_

    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):
        position = inputs.position.squeeze(2).clamp(0, self.num_positions)
        logits = self.embedding(position).squeeze(2)
        return logits

    def forward_pass(self, inputs, features, inputs_mask=None,
                     targets_mask=None):
        return features

    def compute_energy(self, inputs, features, targets, inputs_mask=None,
                       targets_mask=None):

        if targets_mask is None:
            targets_mask = inputs.embedding[:,:,0].eq(self.mask_value)

        pos_probs = torch.sigmoid(features)
        pos_energy = -targets * pos_probs
        neg_probs = 1 - pos_probs
        neg_energy = -(1 - targets) * neg_probs
        pointwise_energy = (pos_energy + neg_energy).masked_fill(
            targets_mask, 0)

        length = Variable(inputs.length.data.float().view(-1, 1))
        total_energy = pointwise_energy.sum(1, keepdim=True)
        mean_energy = total_energy / length
        return mean_energy
