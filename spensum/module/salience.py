from .spen_module import SpenModule
from ntp.modules import MultiLayerPerceptron
import torch.nn as nn


class Salience(SpenModule):
    def __init__(self, embedding_size, hidden_layer_sizes=None, 
                 hidden_layer_activations="tanh", hidden_layer_dropout=0.0,
                 mode="spen", mask_value=-1):
        super(Salience, self).__init__(mode=mode, mask_value=mask_value)

        self.embedding_size_ = embedding_size
        self.mlp_ = MultiLayerPerceptron(
            embedding_size, 1,
            hidden_sizes=hidden_layer_sizes,
            hidden_layer_dropout=hidden_layer_dropout,
            hidden_layer_activations=hidden_layer_activations,
            output_activation="sigmoid")

    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def mlp(self):
        return self.mlp_

    def compute_energy(self, inputs, labels, mask):
        raise NotImplementedError(
            "WordCount module does not yet implement compute_energy")

    def feed_forward(self, inputs, mask):

        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)

        embedding_flat = inputs.embedding.view(
            batch_size * input_size, self.embedding_size)

        prob_flat = self.mlp_(embedding_flat)
        prob = prob_flat.view(batch_size, input_size).masked_fill(mask, 0)
        return prob
