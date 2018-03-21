import ntp
import torch
import torch.nn as nn
import torch.nn.functional as F
from spensum.functional import sequence_dropout
from torch.nn.modules.distance import CosineSimilarity as cosim
import numpy as np

class RNNExtractor(nn.Module):
    def __init__(self, embedding_size, hidden_size, rnn_cell="lstm", layers=1,
                 bidirectional=True, merge_mode="concat"):
        
        #, input_module, encoder_module, predictor_module,
        #         pad_value=0):
        super(RNNExtractor, self).__init__()

        self.rnn_ = ntp.modules.EncoderRNN(
            embedding_size, hidden_size, rnn_cell=rnn_cell, layers=layers,
            bidirectional=bidirectional, merge_mode=merge_mode)
        
        self.predictor_module_ = ntp.modules.MultiLayerPerceptron(
            self.rnn.output_size, 1, output_activation="sigmoid") 
        #self.input_module_ = input_module
        #self.encoder_module_ = encoder_module
        #self.predictor_module_ = predictor_module
        #self.pad_value_ = pad_value


    def forward(self, inputs, mask=None):

        input_embeddings = inputs.embedding.transpose(1, 0)
        input_lengths = inputs.length
        sequence_size = input_embeddings.size(0)
        batch_size = input_embeddings.size(1)

        input_embeddings = sequence_dropout(
            input_embeddings, p=.25, training=self.training)

        context_sequence = self.rnn.encoder_context(
            input_embeddings, length=input_lengths)

        mask = context_sequence[:,:,0].eq(0).transpose(1, 0)

        context_sequence = sequence_dropout(
            context_sequence, p=.25, training=self.training)
        
        context_flat = context_sequence.view(sequence_size * batch_size, -1)
        logits_flat = self.predictor_module(context_flat)
        logits = logits_flat.view(sequence_size, batch_size).transpose(1, 0)
        logits = logits.masked_fill(mask, 0)
        return logits
        
    @property
    def rnn(self):
        return self.rnn_

    @property
    def predictor_module(self):
        return self.predictor_module_

    '''
    Rescores according to similarity of sentence with the query
    '''
    def rescore(self, inputs, probs, bins=[-0.4,-0.1,0.1,0.4], qweight=10.0):
      batches = inputs.embedding.data.size(0)
      assert(probs.data.size(0) == batches)

      embedding = inputs.embedding.data
      # narrow the input tensor
      sen_embeds = embedding.narrow(2, 0, 300)
      query_embeds = embedding.narrow(2, 300, 300)

      # compute the cosine similarity
      cos = cosim(dim=2)
      cos_scores = cos(sen_embeds, query_embeds)

      # normalize the original scores
      sums = probs.data.sum(1)
      divs = sums.repeat(probs.size(1), 1).t()
      probs.data.div_(divs)

      # apply rescoring
      for b in range(0, batches):
        probs.data[b] = probs.data[b] + torch.log(cos_scores[b] + 1.0 + cos.eps)

      # sort to get scores and indices (descending)
      return torch.sort(probs, 1, descending=True)

    def extract(self, inputs, metadata, strategy="rank", word_limit=100, rescore=False):
        probs = self.forward(inputs)
        summaries = []
        if strategy == "rank":
            scores, indices = torch.sort(probs, 1, descending=True)
            if rescore: scores, indices = self.rescore(inputs, probs)
            for b in range(probs.size(0)):
                words = 0
                lines = []
                for i in range(probs.size(1)):
                    idx = indices.data[b][i]
                    if b >= len(metadata.text):
                      print("DEBUG: b=%d text len=%d" % (b, len(metadata.text)))
                    elif idx >= len(metadata.text[b]):
                        print("DEBUG: idx=%d text len2=%d, b=%d" % (idx, len(metadata.text[b]), b))
                        for j in range(5): print(metadata.text[b][j])
                    lines.append(metadata.text[b][idx])
                    words += inputs.word_count.data[b,idx,0]
                    if words >= word_limit:
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
