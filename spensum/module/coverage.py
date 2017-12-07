from .spen_module import SpenModule
from .centroid_mixin import CentroidMixIn
from ntp.modules import MultiLayerPerceptron, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Coverage(SpenModule, CentroidMixIn):
    ''' 
    Coverage score of sentence i is y_i = sigmoid(u * (x_i^T W x_d) + b)
    where x_i is the column vector corresponding to the sentence embedding
    for sentence i, x_d is the average of all sentence emeddings from the input
    instance, W is an embedding size by embedding size matrix of learned 
    parameters, and u and b are learned scalar parameters.
    '''
    def __init__(self, embedding_size, group_dropout=0.0, 
                 mode="spen", mask_value=-1):
        super(Coverage, self).__init__(mode=mode, mask_value=mask_value)
        CentroidMixIn.__init__(self)
       
        self.embedding_size_ = embedding_size
        self.input_layer_norm_ = LayerNorm(embedding_size)
        self.group_layer_norm_ = LayerNorm(embedding_size)
        self.weights = nn.Parameter(
            torch.FloatTensor(embedding_size, embedding_size))
        self.weights.data.normal_(0, 0.00001)
        
        self.sigmoid_layer_ = MultiLayerPerceptron(
            1, 1, output_activation="sigmoid")
        self.group_dropout_ = group_dropout
        
    @property
    def group_dropout(self):
        return self.group_dropout_

    @property
    def embedding_size(self):
        return self.embedding_size_


    def compute_energy(self, inputs, targets, mask):
        batch_size = targets.size(0)
            
        salience = self.feed_forward(inputs, mask)
        non_salience = 1 - salience
        energy = salience.mul(targets.masked_fill(mask, 0)) + \
            non_salience.mul((1 - targets).masked_fill(mask, 0))
        # TODO make this a masked mean.
        avg_energy = energy.mean(1, keepdim=True)
        return avg_energy



    # TODO should really just create a sequence bilinear layer
    def feed_forward(self, inputs, mask):
        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)

        centroid = self.centroid(inputs, mask, remove_identity=True)
        centroid = self.group_layer_norm_(centroid)
        centroid_flat = centroid.view(batch_size * input_size, -1, 1)

        embedding = self.input_layer_norm_(inputs.embedding)
        embedding_flat = embedding.view(batch_size * input_size, 1, -1)
        
        weights = self.weights.view(
            1, self.embedding_size, self.embedding_size).repeat(
            batch_size * input_size, 1, 1)

        emb_flat_proj = embedding_flat.bmm(weights)
        dotsim_flat = emb_flat_proj.bmm(centroid_flat).squeeze(2)

        output_flat = self.sigmoid_layer_(dotsim_flat)
        output = output_flat.view(batch_size, input_size).masked_fill(mask, 0)
        return output
