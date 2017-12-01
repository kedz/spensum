from ntp.modules import MultiLayerPerceptron, LayerNorm
import torch
import torch.nn as nn
from torch.autograd import Variable

class BinaryGroupSalience(nn.Module):
    def __init__(self, input_size, interaction_mode="concat",
                 hidden_layer_sizes=None, hidden_layer_activations="tanh", 
                 hidden_layer_dropout=0.0,
                 mask_value=-1):
        super(BinaryGroupSalience, self).__init__()

        self.mask_value = mask_value
        self.interaction_mode_ = interaction_mode
       
        self.input_layer_norm_ = LayerNorm(input_size)
        self.group_layer_norm_ = LayerNorm(input_size)
        
        if interaction_mode == "concat":
            input_size *= 2


        self.mlp_ = MultiLayerPerceptron(
            input_size,
            1,
            hidden_sizes=hidden_layer_sizes,
            hidden_layer_dropout=hidden_layer_dropout,
            hidden_layer_activations=hidden_layer_activations,
            output_activation="sigmoid")

    @property
    def interaction_mode(self):
        return self.interaction_mode_

    def prepare_inputs(self, inputs):
        num_inputs = inputs.size(0)
        max_len = inputs.size(1)
        emb_size = inputs.size(2)

        mask = inputs.data.eq(self.mask_value)
        inputs_masked = inputs.masked_fill(mask, 0).data
        counts = max_len - mask.float().sum(1) - 1
        group_mean = inputs_masked.sum(1) / counts
        
        over_count = inputs_masked / counts.unsqueeze(1).repeat(
            1, max_len, 1)
        group_inputs = group_mean.unsqueeze(1).repeat(1, max_len, 1) \
            - over_count
        
        group_inputs_masked = group_inputs.masked_fill_(mask, 0)

        inputs_masked = self.input_layer_norm_(
            Variable(inputs_masked))
        group_inputs_masked = self.input_layer_norm_(
            Variable(group_inputs_masked))


        if self.interaction_mode == "concat":
            prepared_inputs = torch.cat(
                [inputs_masked, group_inputs_masked], 2)
        else:
            prepared_inputs = inputs_masked + group_inputs_masked

        return prepared_inputs

    def forward(self, inputs):
        mask = inputs.data[:,:,0].eq(self.mask_value)

        inputs = self.prepare_inputs(inputs)

        num_inputs = inputs.size(0)
        max_len = inputs.size(1)
        emb_size = inputs.size(2)

        inputs_flat = inputs.view(num_inputs * max_len, emb_size)
        output_flat = self.mlp_(inputs_flat)
        output = output_flat.view(num_inputs, max_len)
        output.data.masked_fill_(mask, 0)

        return output
