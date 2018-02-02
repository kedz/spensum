import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from spensum.functional import sequence_dropout
from spensum.model.sequence_standardizer import SequenceStandardizer


class RNNModel(nn.Module):
    def __init__(self, embedding_size=300, hidden_size=300, dropout=.5,
                 standardize_input=True):
    
        super(RNNModel, self).__init__()

        self.dropout = dropout
        self.rnn = nn.GRU(
            embedding_size, hidden_size, bidirectional=True, num_layers=1)

        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

        if standardize_input:
            self.standardizer = SequenceStandardizer(embedding_size)
        else:
            self.standardizer = None

    def forward(self, inputs):
        batch_size = inputs.sequence.size(0)
        seq_size = inputs.sequence.size(1)

        if self.standardizer is not None:
            inputs_sequence = self.standardizer(inputs)
        else:
            inputs_sequence = inputs.sequence

        inputs_sequence = sequence_dropout(
            inputs_sequence, p=self.dropout, training=self.training, 
            batch_first=True)

        packed_input = nn.utils.rnn.pack_padded_sequence(
                inputs.sequence, inputs.length.data.tolist(), 
                batch_first=True)
        packed_context, _ = self.rnn(packed_input)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
                packed_context, batch_first=True)

        mlp_layer1 = sequence_dropout(
            F.relu(self.linear1(hidden_states)), p=self.dropout,
            training=self.training, batch_first=True)
        mlp_layer2 = self.linear2(mlp_layer1).view(batch_size, -1)

        return mlp_layer2

    def predict(self, inputs, metadata, max_words=100, return_indices=False):

        mask = inputs.sequence.data[:,:,0].eq(-1)
        batch_size = inputs.sequence.size(0)

        logits = self.forward(inputs).data.masked_fill_(mask, float("-inf"))
        _, indices = torch.sort(logits, 1, descending=True)

        all_pos = []
        all_text = []
        for b in range(batch_size):
            wc = 0
            text = []
            pos = [] 
            for i in indices[b]:
                if i >= inputs.length.data[b]:
                    break
                text.append(metadata.text[b][i])
                pos.append(i)
                wc += inputs.word_count.data[b,i]

                if wc > max_words:
                    break
            all_pos.append(pos)
            all_text.append(text)
        if return_indices:
            return all_text, all_pos
        else:
            return all_text
