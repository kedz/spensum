import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SpenModule(nn.Module, ABC):

    def __init__(self, mode="spen", mask_value=-1):
        super(SpenModule, self).__init__()
        self.mode = mode
        self.mask_value=mask_value

    def spen(self):
        self.mode = "spen"

    def pretrain(self):
        self.mode = "pretrain"

    @property
    def mode(self):
        return self.mode_

    @mode.setter
    def mode(self, new_mode):
        if new_mode not in ["spen", "pretrain"]:
            raise Exception("mode must be either 'spen' or 'pretrain'")
        self.mode_ = new_mode

    def forward(self, inputs, labels=None, mask=None):
        if mask is None:
            mask = inputs.embedding[:,:,0].eq(self.mask_value)
        if self.mode == "spen":
            return self.compute_energy(inputs, labels, mask)
        else:
            return self.feed_forward(inputs, mask)

    def compute_energy(self, inputs, targets, mask):
        batch_size = targets.size(0)
            
        potential_energy = self.feed_forward(inputs, mask)
        energy = torch.bmm(
            targets.view(batch_size, 1, -1),
            potential_energy.view(batch_size, 1, -1).permute(0,2,1))
        energy = energy.view(-1, 1)
        return energy

    @abstractmethod
    def feed_forward(self, inputs, mask):
        pass

