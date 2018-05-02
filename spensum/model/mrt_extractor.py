import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from spensum.criterion.rouge_score import eval_rouge
from spensum.model.rnn_extractor import RNNExtractor
import ntp
import math

class MRTExtractor(RNNExtractor):

  def __init__(self, embedding_size, hidden_size, refs_dict, num_samples = 5, budget = 3, alpha = 2.0, rnn_cell="lstm", layers=1, bidirectional=True, merge_mode="concat", gamma = 0.96, stopwords = set()):
    super(MRTExtractor, self).__init__(embedding_size, hidden_size, rnn_cell=rnn_cell, layers=layers,
                 bidirectional=bidirectional, merge_mode=merge_mode)
    self.num_samples = num_samples
    self.budget = budget
    self.refs_dict = refs_dict
    self.alpha = alpha
    self.gamma = gamma
    self.stopwords = stopwords

  def get_risk(self, samples, ids, texts, stopwords):
        """
        Selects the first *budget* sentences that are equal to 1 and computes
        the total risk for that selection.
        Returns a batch_size x sample_size tensor of risks for each sample.
        In this callback you would implement 1 - Rouge for each sample.
        """

        batch_size = samples.size(0)
        sample_size = samples.size(1)
        seq_size = samples.size(2)
        sample_risk = samples.data.new(
            batch_size, sample_size).float().fill_(0)

        for b in range(batch_size):
            for s in range(sample_size):
                count = 0 
                tokens = []
                for sen in range(seq_size):
                    if samples.data[b, s, sen] == 1 and sen < len(texts[b]):
                        tmp = texts[b][sen].split(" ")
                        count += len(tmp)
                        if len(tokens) + len(tmp) <= self.budget:
                          tokens.extend(tmp)
                # this is the computation of risk based on rouge
                penalty = math.pow(self.gamma,max(1.0,count-self.budget))
                sample_risk[b, s] = 1.0 - eval_rouge([tokens],[self.refs_dict[ids[b]]],self.stopwords)*penalty
        return Variable(sample_risk)

  def make_mask(self, lengths):
    max_len = lengths.data.max()
    mask = lengths.data.new(lengths.size(0), max_len).fill_(0)

    for i, l in enumerate(lengths.data.tolist()):
        if l < max_len:
            mask[i,l:].fill_(1)
    return Variable(mask.byte())

  def forward_mrt(self, inputs, metadata):
    logits = super(MRTExtractor, self).forward(inputs)
    lengths = inputs.length
    mask = self.make_mask(lengths)

    sample_mask = mask.unsqueeze(1).repeat(1, self.num_samples, 1)

    ### Compute the model distribution P ###

    # probs is a batch_size x num_samples x seq_size tensor.
    probs = F.sigmoid(logits)
    probs = probs.unsqueeze(1).repeat(1, self.num_samples, 1)

    # overcomplete_probs is a batch_size x num_samples x seq_size x 2 tensor 
    # of probs where the last dim has probs of 0 and 1. Last dim sums to 1.
    # We need this to efficiently index the probabilities of each sentence
    # selection decision.
    probs4D = probs.unsqueeze(3)
    overcomplete_probs = torch.cat([1 - probs4D, probs4D], 3)

    # Samples is a batch_size x num_samples x seq_size tensor of 0s and 1s.
    # 1 indicates we are selecting the corresponding sentence to go in the
    # summary. samples is cast to a long tensor so we can use it to efficiently
    # index the probability of each decision so we can compute the likelihood
    # of each sequence under the model distribution P.
    samples = torch.bernoulli(probs)
    samples = Variable(samples.data.long())

    # Get probability of each sentence decision
    stepwise_sample_probs = overcomplete_probs.gather(
        dim=3, index=samples.unsqueeze(3)).squeeze(3)
    # Compute the sample log likelihood under the model distribution P:
    # log P(y|x) = sum logP(y_1|x) + logP(y_2|x) ...
    # sample_log_likelihood is a batch_size x sample_size tensor.
    sample_log_likelihood = torch.log(stepwise_sample_probs).masked_fill(
        sample_mask, 0).sum(2)
    
    # Length normalize the log likelihoods and take the softmax over each 
    # sample to get the approximate distribution Q.
    # For k samples:
    #
    #                       exp(logP(y^(i)|x)) 
    #  Q(y^(i)|x) =  -----------------------------------------
    #                  sum_j=1^k exp(logP(y^(j)|x))
    #
    q_logits = sample_log_likelihood / lengths.float().view(-1, 1) * self.alpha
    q_probs = F.softmax(q_logits, 1)

    # Here is where you would call your rouge function and return 1 - rouge
    # to make it a risk. sample_risks is a batch_size x num_samples tensor.
    sample_risks = self.get_risk(samples, metadata.id, metadata.text)
    expected_risk = (q_probs * sample_risks).sum(1)
    # average the expected_risk over the batch
    avg_expected_risk = expected_risk.mean()
    return avg_expected_risk
