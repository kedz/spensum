from ntp.modules import MultiLayerPerceptron, LayerNorm

import torch
import torch.nn as nn
from torch.autograd import Variable
from .binary_salience import BinarySalience


class Novelty(nn.Module):
    def __init__(self, input_size, salience_model, 
                 freeze_salience_module=False, mask_value=-1):
        super(Novelty, self).__init__()
        self.input_size_ = input_size
        self.salience_model_ = salience_model
        self.input_layer_norm_ = LayerNorm(input_size)
        self.group_layer_norm_ = LayerNorm(input_size)
        self.mask_value_ = mask_value

        self.weights = nn.Parameter(
            torch.diag(torch.FloatTensor(input_size).fill_(.001)).float())
        self.sigmoid_layer = MultiLayerPerceptron(
            1, 1, output_activation="sigmoid")

        self.freeze_salience_module = freeze_salience_module

    def parameters(self):
        if not self.freeze_salience_module:
            for param in self.salience_model.parameters():
                yield param
        for param in self.input_layer_norm_.parameters():
            yield param
        for param in self.group_layer_norm_.parameters():
            yield param
        yield self.weights
        for param in self.sigmoid_layer.parameters():
            yield param

    @property
    def mask_value(self):
        return self.mask_value_ 

    @property
    def input_size(self):
        return self.input_size_

    @property
    def salience_model(self):
        return self.salience_model_

    @salience_model.setter
    def salience_model(self, new_module):
        if not isinstance(new_module, BinarySalience):
            raise Exception("Must be a BinarySalience module.")
        self.salience_model_ = new_module

    # TODO preplace coverage prepate inputs with this method
    def init_identity_mask(self, batch_size, doc_size):
        ''' 
        Compute a mask doc_size x doc_size mask where all values are 1 
        except along the diagonal which is 0. 
        This mask is repeated batch_size times to create a 
        batch_size x doc_size x doc_size mask tensor.
        '''
        mask = torch.FloatTensor(doc_size, doc_size).fill_(1) - \
            torch.diag(torch.FloatTensor(doc_size).fill_(1))
        mask_tensor = mask.view(1, doc_size, doc_size).repeat(batch_size, 1, 1)
        return Variable(mask_tensor)

    def prepare_inputs(self, inputs):
        batch_size = inputs.size(0)
        doc_size = inputs.size(1)

        mask = inputs.data.eq(self.mask_value)
        inputs_masked = inputs.masked_fill(mask, 0).data

        identity_mask = self.init_identity_mask(batch_size, doc_size)
        salience = self.salience_model(inputs)
        batch_weights = salience.view(batch_size, 1, doc_size).repeat(
            1, doc_size, 1)
        batch_weights = batch_weights.mul(identity_mask)
        group_inputs = batch_weights.bmm(inputs)
        #group_inputs_masked = group_inputs.masked_fill_(mask, 0)
     
        inputs_masked = self.input_layer_norm_(
            Variable(inputs_masked))
        group_inputs = self.input_layer_norm_(
            group_inputs)

        return inputs_masked, group_inputs


    def forward(self, inputs):
        mask = inputs.data[:,:,0].eq(self.mask_value)
        batch_size = inputs.size(0)
        group_len = inputs.size(1)
        emb_size = inputs.size(2)

        inputs, groups = self.prepare_inputs(inputs)
        #if self.group_dropout > 0:
        #    groups = F.dropout(
        #        groups, p=self.group_dropout, training=self.training)
        weights = self.weights.view(
            1, self.input_size, self.input_size).repeat(
            batch_size * group_len, 1, 1)

        inputs_flat = inputs.view(batch_size * group_len, 1, emb_size)
        groups_flat = groups.view(batch_size * group_len, emb_size, 1)
        inputs_flat_proj = inputs_flat.bmm(weights)
        coverage_flat = inputs_flat_proj.bmm(groups_flat).squeeze(2)
        summary_coverage = self.sigmoid_layer(coverage_flat).view(
            batch_size, group_len)
        novelty = 1 - summary_coverage
        return novelty.masked_fill_(Variable(mask), 0)

