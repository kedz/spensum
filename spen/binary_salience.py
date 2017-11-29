from ntp.modules import MultiLayerPerceptron
import torch.nn as nn


class BinarySalience(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, 
                 hidden_layer_activations="tanh", hidden_layer_dropout=0.0):
        super(BinarySalience, self).__init__()

        self.mlp_ = MultiLayerPerceptron(
            input_size,
            1,
            hidden_sizes=hidden_sizes,
            hidden_layer_dropout=hidden_layer_dropout,
            hidden_layer_activations=hidden_layer_activations,
            output_activation="sigmoid")

    def forward(self, inputs):

        num_inputs = inputs.size(0)
        max_len = inputs.size(1)
        emb_size = inputs.size(2)
    
        input_flat = inputs.view(num_inputs * max_len, emb_size)
        output_flat = self.mlp_(input_flat)
        output = output_flat.view(num_inputs, max_len)
        mask = inputs.data[:,:,0].eq(-1)
        output.data.masked_fill_(mask, 0)
        return output
