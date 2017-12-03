from ntp.modules import MultiLayerPerceptron
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class JointModel(nn.Module):
    def __init__(self, energy_modules, mask_value=-1):
        super(JointModel, self).__init__()
        self.energy_modules_ = nn.ModuleList(energy_modules)
        self.module_interpolation = MultiLayerPerceptron(
            len(energy_modules), 1, output_activation="sigmoid",
            hidden_sizes=[20, 20],
            hidden_layer_activations="relu",
            hidden_layer_dropout=.5)
        self.mask_value_ = mask_value

    @property
    def mask_value(self):
        return self.mask_value_
    
    @property
    def energy_modules(self):
        return self.energy_modules_

    def forward(self, inputs):
        mask = inputs.embedding[:,:,0].eq(self.mask_value)
        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)
        
        flat_module_outputs = []
        for mod in self.energy_modules:
            module_output = mod(inputs, mask=mask)
            flat_module_outputs.append(
                module_output.view(batch_size * input_size, 1))
        flat_module_outputs = torch.cat(flat_module_outputs, 1)
        prob_flat = self.module_interpolation(flat_module_outputs)
        prob = prob_flat.view(batch_size, input_size).masked_fill(mask, 0)

        return prob
