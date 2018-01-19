import torch
import torch.nn as nn
from abc import ABC, abstractmethod

def _update_counter(self, grad_input, grad_output):
    self._burn_counter += 1

class SpenModule(nn.Module, ABC):

    def __init__(self, name="SpenModule", burn_in=0, mask_value=-1):
        super(SpenModule, self).__init__()
        self.name_ = name
        self.mask_value_ = mask_value

        self._burn_counter = 0
        self._burn_in = burn_in
        self.register_backward_hook(_update_counter)

    @property
    def ready(self):
        return self.burn_in <= self._burn_counter

    @property
    def burn_in(self):
        return self._burn_in

    @property
    def burn_in_iters(self):
        return max(0, self.burn_in - self._burn_counter)

    @property
    def name(self):
       return self.name_

    @property
    def mask_value(self):
        return self.mask_value_

    @abstractmethod
    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):
        pass

    @abstractmethod
    def forward_pass(self, inputs, features, inputs_mask=None, 
                     targets_mask=None):
        pass

    @abstractmethod
    def compute_energy(self, inputs, features, targets, inputs_mask=None, 
                       targets_mask=None):
        pass


    def forward(self, inputs, targets=None, precomputed=None,
                inputs_mask=None, targets_mask=None):

        if precomputed is None:
            features = self.compute_features(
                inputs, inputs_mask=inputs_mask, targets_mask=targets_mask)
        else:
            features = precomputed

        if targets is None:
            output = self.forward_pass(
                inputs, features, inputs_mask=inputs_mask, 
                targets_mask=targets_mask)
            return output

        else:
            energy = self.compute_energy(
                inputs, features, targets, inputs_mask=inputs_mask,
                targets_mask=targets_mask)
            return energy




#        if mask is None:
#            mask = inputs.embedding[:,:,0].eq(self.mask_value)
#        if self.mode == "spen":
#            return self.compute_energy(inputs, labels, mask)
#        else:
#            return self.feed_forward(inputs, mask)
#
#    def compute_energy(self, inputs, targets, mask):
#        batch_size = targets.size(0)
#            
#        potential_energy = self.feed_forward(inputs, mask)
#        energy = torch.bmm(
#            targets.view(batch_size, 1, -1),
#            potential_energy.view(batch_size, 1, -1).permute(0,2,1))
#        energy = energy.view(-1, 1)
#        return energy
#
#    @abstractmethod
#    def feed_forward(self, inputs, mask):
#        pass

