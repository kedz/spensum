from .spen_module import SpenModule
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class WordCount(SpenModule):
    def __init__(self, name="WordCount", burn_in=0, mask_value=-1):
        super(WordCount, self).__init__(
            name=name, burn_in=burn_in, mask_value=mask_value)
        self.weight = nn.Parameter(torch.randn(1, 1))
        self.bias = nn.Parameter(torch.ones(1, 1))

    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):
        word_count = inputs.word_count.squeeze(2)
        logits = word_count * self.weight + self.bias
        return logits

    def forward_pass(self, inputs, features, inputs_mask=None,
                     targets_mask=None):

        if targets_mask is None:
            targets_mask = inputs.embedding[:,:,0].eq(self.mask_value)
        return features.masked_fill(targets_mask, 0)

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
