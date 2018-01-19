from .spen_module import SpenModule
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import ntp
from spensum.functional import sequence_dropout



class PCCoverage(SpenModule):

    def __init__(self, embedding_size, num_components=3, name="PCCoverage", 
                 input_hidden_layer_sizes=[100,100], 
                 input_hidden_layer_activation="relu",
                 input_hidden_layer_dropout=.5,
                 input_dropout=.5,
                 pc_dropout=.5,
                 mask_value=-1, burn_in=0):

        super(PCCoverage, self).__init__(
            name=name, mask_value=mask_value, burn_in=burn_in)
        self.input_mlp = ntp.modules.MultiLayerPerceptron(
            embedding_size, num_components * 100, output_activation="relu",
            hidden_sizes=input_hidden_layer_sizes,
            hidden_layer_activations=input_hidden_layer_activation,
            hidden_layer_dropout=input_hidden_layer_dropout)
        self.pc_mlps = nn.ModuleList(
            [ntp.modules.MultiLayerPerceptron(
                    embedding_size, 100, hidden_sizes=[150],
                    hidden_layer_activations="relu",
                    hidden_layer_dropout=.5,
                    output_activation="relu")
             for i in range(num_components)])
        self.input_dropout = input_dropout
        self.pc_dropout = pc_dropout
        
    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):

        self.input_mlp.output_size

        if targets_mask is None:
            targets_mask = inputs.embedding[:,:,0:1].eq(
                self.mask_value).repeat(1, 1, self.input_mlp.output_size)
        else:
            targets_mask = targets_mask.unsqueeze(2).repeat(
                1, 1, self.input_mlp.output_size)
        
        inputs_embedding = sequence_dropout(
                inputs.embedding, p=self.input_dropout,
                training=self.training, batch_first=True)
        pc_saliences = []
        for i, pc_mlp in enumerate(self.pc_mlps):
            pc = F.dropout(
                inputs.principal_components[:,i], p=self.pc_dropout,
                training=self.training)
            pc_saliences.append(pc_mlp(pc))
        pc_saliences = torch.cat(pc_saliences, 1)

        batch_size = inputs.embedding.size(0)
        seq_size = inputs.embedding.size(1)
        inputs_embedding_flat = inputs_embedding.view(
            batch_size * seq_size, -1)

        input_saliences = self.input_mlp(inputs_embedding_flat).view(
            batch_size, seq_size, -1).masked_fill(targets_mask, 0)

        coverage = input_saliences.bmm(pc_saliences.view(batch_size, -1, 1))
        coverage = coverage.view(batch_size, -1)
        return coverage

    def forward_pass(self, inputs, features, inputs_mask=None,
                     targets_mask=None):

        return features       

    def compute_energy(self, inputs, features, targets, inputs_mask=None,
                       targets_mask=None):

        if targets_mask is None:
            targets_mask = inputs.embedding[:,:,0].eq(self.mask_value)

        pos_probs = torch.sigmoid(features)
        pos_energy = -targets * pos_probs
        neg_probs = 1 - pos_probs
        neg_energy = -(1 - targets) * neg_probs
        pointwise_energy = (pos_energy + neg_energy).masked_fill(
            targets_mask, 0)

        length = Variable(inputs.length.data.float().view(-1, 1))
        total_energy = pointwise_energy.sum(1, keepdim=True)
        mean_energy = total_energy / length
        return mean_energy

    def extract(self, inputs, metadata, strategy="rank", word_limit=100):

        logits = self.forward(inputs)
        probs = torch.sigmoid(logits)
        probs = probs.masked_fill(
            inputs.embedding[:,:,0].eq(self.mask_value), 0)
       


        summaries = []
        if strategy == "rank":
            scores, indices = torch.sort(probs, 1, descending=True)
            for b in range(probs.size(0)):
                words = 0
                lines = []
                for i in range(probs.size(1)):
                    idx = indices.data[b][i]
                    if idx < len(metadata.text[b]):
                        lines.append(metadata.text[b][idx])
                        words += inputs.word_count.data[b,idx,0]
                        if words > word_limit:
                            break
                summaries.append("\n".join(lines))
        elif strategy == "in-order":
            for b in range(probs.size(0)):
                words = 0
                lines = []
                for i in range(probs.size(1)):
                    if probs.data[b][i] > .5:
                        lines.append(metadata.text[b][i])
                        words += inputs.word_count.data[b,i,0]
                        if words > word_limit:
                            break
                summaries.append("\n".join(lines))
        else:
            raise Exception("strategy must be 'rank' or 'in-order'")
        return summaries

