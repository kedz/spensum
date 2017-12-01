from .spen_module import SpenModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalSalience(SpenModule):
    def __init__(self, embedding_size, num_positions, mode="spen", 
                 mask_value=-1):
        super(PositionalSalience, self).__init__(mode=mode, 
                                                 mask_value=mask_value)

        self.embedding_size_ = embedding_size
        self.num_positions_ = num_positions
        self.weight = nn.Embedding(
            num_positions + 1, embedding_size, padding_idx=0)
        self.bias = nn.Embedding(
            num_positions + 1, 1, padding_idx=0)
        
    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def num_positions(self):
        return self.num_positions_

    def compute_energy(self, inputs, labels, mask):
        raise NotImplementedError(
            "Positional Salience module does not yet implement compute_energy")

    def feed_forward(self, inputs, mask):
        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)
        flat_size = batch_size * input_size
        
        position = inputs.position.squeeze(2).clamp(0, self.num_positions)

        weight_flat = self.weight(position.view(flat_size, 1))
        i_emb_flat = inputs.embedding.view(flat_size, -1, 1)
        bias_flat = self.bias(position.view(flat_size))
        
        logits_flat = weight_flat.bmm(i_emb_flat).squeeze(2) + bias_flat
        logits = logits_flat.view(batch_size, input_size)
        
        output = F.sigmoid(logits).masked_fill(mask, 0)
        return output
