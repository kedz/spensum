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


    @abstractmethod
    def compute_energy(self, inputs, labels, mask):
        pass

    @abstractmethod
    def feed_forward(self, inputs, mask):
        pass
