from .spen_module import SpenModule
import torch.nn as nn
import torch.nn.functional as F


class Position(SpenModule):
    def __init__(self, num_positions, mode="spen", mask_value=-1):
        super(Position, self).__init__(mode=mode, mask_value=mask_value)
        self.num_positions_ = num_positions
        self.embedding = nn.Embedding(num_positions + 1, 1)
    
    @property
    def num_positions(self):
        return self.num_positions_

    def compute_energy(self, inputs, targets, mask):
        batch_size = targets.size(0)
            
        on_position = self.feed_forward(inputs, mask)
        off_position = 1 - on_position
        energy = on_position.mul(targets.masked_fill(mask, 0)) + \
            off_position.mul((1 - targets).masked_fill(mask, 0))
        # TODO make this a masked mean.
        avg_energy = energy.mean(1, keepdim=True)
        return avg_energy

    def feed_forward(self, inputs, mask):
        position = inputs.position.squeeze(2).clamp(0, self.num_positions)
        logits = self.embedding(position).squeeze(2)
        prob = F.sigmoid(logits).masked_fill(mask, 0)
        return prob
