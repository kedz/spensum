from .spen_module import SpenModule
import torch
from torch.autograd import Variable
import torch.nn as nn
import ntp
from spensum.functional import sequence_dropout


class RNNSalience(SpenModule):
    def __init__(self, embedding_size, hidden_size=None, rnn_cell="lstm",
                 layers=2, bidirectional=True, merge_mode="concat",
                 embedding_dropout=.5,
                 rnn_dropout=.5,
                 mask_value=-1, 
                 burn_in=0, name="RNNSalience"):
        
        super(RNNSalience, self).__init__(
            name=name, burn_in=burn_in, mask_value=mask_value)

        if hidden_size is None:
            hidden_size = embedding_size

        self.rnn = ntp.modules.EncoderRNN(
            embedding_size, hidden_size, rnn_cell=rnn_cell, layers=layers,
            bidirectional=bidirectional, merge_mode=merge_mode)
        
        self.mlp = ntp.modules.MultiLayerPerceptron(
            self.rnn.output_size, 1, output_activation=None)

        self.embedding_dropout = embedding_dropout
        self.rnn_dropout = rnn_dropout
        

    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):

        input_embeddings = sequence_dropout(
            inputs.embedding.transpose(1,0),
            p=self.embedding_dropout,
            training=self.training)

        sequence_size = input_embeddings.size(0)
        batch_size = input_embeddings.size(1)

        context_sequence = self.rnn.encoder_context(
            input_embeddings, length=inputs.length)

        context_sequence = sequence_dropout(
            context_sequence, p=self.rnn_dropout, training=self.training)

        context_flat = context_sequence.view(sequence_size * batch_size, -1)
        logits_flat = self.mlp(context_flat)
        logits = logits_flat.view(sequence_size, batch_size).transpose(1, 0)

        return logits

    def forward_pass(self, inputs, features, inputs_mask=None, 
                     targets_mask=None):

        if targets_mask is None:
            targets_mask = inputs.embedding[:,:,0].eq(self.mask_value)

        output = features.masked_fill(targets_mask, 0)

        return output

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


#?    def extract(self, inputs, metadata, strategy="rank", word_limit=100):
#?
#?        probs = self.forward(inputs)
#?        summaries = []
#?        if strategy == "rank":
#?            scores, indices = torch.sort(probs, 1, descending=True)
#?            for b in range(probs.size(0)):
#?                words = 0
#?                lines = []
#?                for i in range(probs.size(1)):
#?                    idx = indices.data[b][i]
#?                    if idx < len(metadata.text[b]):
#?                        lines.append(metadata.text[b][idx])
#?                        words += inputs.word_count.data[b,idx,0]
#?                        if words > word_limit:
#?                            break
#?                summaries.append("\n".join(lines))
#?        elif strategy == "in-order":
#?            for b in range(probs.size(0)):
#?                words = 0
#?                lines = []
#?                for i in range(probs.size(1)):
#?                    if probs.data[b][i] > .5:
#?                        lines.append(metadata.text[b][i])
#?                        words += inputs.word_count.data[b,i,0]
#?                        if words > word_limit:
#?                            break
#?                summaries.append("\n".join(lines))
#?        else:
#?            raise Exception("strategy must be 'rank' or 'in-order'")
#?        return summaries

