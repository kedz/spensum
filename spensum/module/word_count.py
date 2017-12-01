from .spen_module import SpenModule
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordCount(SpenModule):
    def __init__(self, mode="spen", mask_value=-1):
        super(WordCount, self).__init__(mode=mode, mask_value=mask_value)
        self.weight_ = nn.Parameter(torch.rand(1))
        self.bias_ = nn.Parameter(torch.ones(1))

    def compute_energy(self, inputs, labels, mask):
        raise NotImplementedError(
            "WordCount module does not yet implement compute_energy")

    def feed_forward(self, inputs, mask):
        word_count = inputs.word_count.squeeze(2)
        batch_size = word_count.size(0)
        input_size = word_count.size(1)
        word_count_flat = word_count.view(batch_size * input_size, 1)
        logit = word_count * self.weight_ + self.bias_
        prob = F.sigmoid(logit).masked_fill(mask, 0)
        return prob 
