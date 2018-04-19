import ntp
import torch
import torch.nn as nn
import torch.nn.functional as F
from spensum.functional import sequence_dropout
from torch.nn.modules.distance import CosineSimilarity as cosim
import numpy as np
import os

class RNNExtractor(nn.Module):
    def __init__(self, embedding_size, hidden_size, rnn_cell="lstm", layers=1,
                 bidirectional=True, merge_mode="concat"):
        
        super(RNNExtractor, self).__init__()

        self.rnn_ = ntp.modules.EncoderRNN(
            embedding_size, hidden_size, rnn_cell=rnn_cell, layers=layers,
            bidirectional=bidirectional, merge_mode=merge_mode)
        
        self.predictor_module_ = ntp.modules.MultiLayerPerceptron(
            self.rnn.output_size, 1, output_activation="sigmoid")
   
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
    def rescore(self, inputs, probs):
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

    def get_scores(self, inputs):
      embedding = inputs.embedding.data
      # narrow the input tensor
      sen_embeds = embedding.narrow(2, 0, 300)
      query_embeds = embedding.narrow(2, 300, 300)
      cos = cosim(dim=2)
      cos_scores = cos(sen_embeds, query_embeds)
      return torch.log(cos_scores + 1.0 + cos.eps)

    def extract(self, inputs, metadata, strategy="rank", word_limit=100, rescore=False, model=None):
        input_scores = self.get_scores(inputs)
        probs = self.forward(inputs)
        # if another model is given, we average the scores
        if model:
          probs2 = model.forward(inputs)
          probs = 0.5*probs + 0.5*probs2
        summaries = []
        scores = []
        _, indices = torch.sort(probs, 1, descending=True)
        if rescore: _, indices = self.rescore(inputs, probs)
        if strategy == "rand":
          indices.data += torch.Tensor(indices.size(0), indices.size(1)).uniform_(0, 10000).long()
          _, indices = torch.sort(indices, 1, descending=False)
        if strategy == "lead3":
          for b in range(indices.size(0)):
            for i in range(indices.size(1)):
              indices.data[b][i] = i

        for b in range(probs.size(0)):
          words = 0
          lines = []
          score = 0.0
          for i in range(probs.size(1)):
            idx = indices.data[b][i]
            candidate = metadata.text[b][idx]
            sen_words = len(candidate.split(" "))
            if words + sen_words <= word_limit:
              lines.append(candidate)
              score += input_scores[b][idx]
              words += sen_words
            if words == word_limit:
              break
                
          scores.append(score / len(lines))
          summaries.append("\n".join(lines))
        return summaries, scores
