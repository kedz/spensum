from ntp.modules import MultiLayerPerceptron, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Coverage(nn.Module):
    ''' 
    Coverage score of sentence i is y_i = sigmoid(u * (x_i^T W x_d) + b)
    where x_i is the column vector corresponding to the sentence embedding
    for sentence i, x_d is the average of all sentence emeddings from the input
    instance, W is an embedding size by embedding size matrix of learned 
    parameters, and u and b are learned scalar parameters.
    '''
    def __init__(self, input_size, learn_params=True, mask_value=-1,
                 group_dropout=0.0):
        super(Coverage, self).__init__()
        self.mask_value = mask_value
       
        self.input_size_ = input_size
        self.input_layer_norm_ = LayerNorm(input_size)
        self.group_layer_norm_ = LayerNorm(input_size)
        self.weights = nn.Parameter(torch.FloatTensor(input_size, input_size))
        self.weights.data.normal_(0, 0.00001)
        
        self.sigmoid_layer_ = MultiLayerPerceptron(
            1, 1, output_activation="sigmoid")
        self.group_dropout_ = group_dropout
        
    @property
    def group_dropout(self):
        return self.group_dropout_

    @property
    def input_size(self):
        return self.input_size_

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

        return inputs_masked, group_inputs_masked

    def forward(self, inputs):

        mask = inputs.data[:,:,0].eq(self.mask_value)
        batch_size = inputs.size(0)
        group_len = inputs.size(1)
        emb_size = inputs.size(2)

        inputs, groups = self.prepare_inputs(inputs)
        if self.group_dropout > 0:
            groups = F.dropout(
                groups, p=self.group_dropout, training=self.training)
        weights = self.weights.view(
            1, self.input_size, self.input_size).repeat(
            batch_size * group_len, 1, 1)

        inputs_flat = inputs.view(batch_size * group_len, 1, emb_size)
        groups_flat = groups.view(batch_size * group_len, emb_size, 1)
        inputs_flat_proj = inputs_flat.bmm(weights)
        coverage_flat = inputs_flat_proj.bmm(groups_flat).squeeze(2)
        squashed_coverage = self.sigmoid_layer_(coverage_flat).view(
            batch_size, group_len)
        squashed_coverage.data.masked_fill_(mask, 0)
        return squashed_coverage 
