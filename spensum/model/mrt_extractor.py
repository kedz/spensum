import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np


def sample_risk(batch_size, lengths, budget):

    max_len = lengths.data.max()
    risk_sample = []
    for b in range(batch_size):
        risks = [-1] * budget + [1] * (lengths.data[b] - budget)
        random.shuffle(risks)
        if lengths.data[b] < max_len:
            risks.extend([0] * (max_len - lengths.data[b]))
        risk_sample.append(risks)
    return np.array(risk_sample)

def make_mask(lengths):
    max_len = lengths.data.max()
    mask = lengths.data.new(lengths.size(0), max_len).fill_(0)

    for i, l in enumerate(lengths.data.tolist()):
        if l < max_len:
            mask[i,l:].fill_(1)
    return Variable(mask.byte())

def forward(logits, mask, lengths, num_samples, risk_callback):
    sample_mask = mask.unsqueeze(1).repeat(1, num_samples, 1)

    ### Compute the model distribution P ###

    # probs is a batch_size x num_samples x seq_size tensor.
    probs = F.sigmoid(logits)
    probs = probs.unsqueeze(1).repeat(1, num_samples, 1)

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
    q_logits = sample_log_likelihood / lengths.float().view(-1, 1)
    q_probs = F.softmax(q_logits, 1)

    # Here is where you would call your rouge function and return 1 - rouge
    # to make it a risk. sample_risks is a batch_size x num_samples tensor.
    sample_risks = risk_callback(samples)
    expected_risk = (q_probs * sample_risks).sum(1)
    # average the expected_risk over the batch
    avg_expected_risk = expected_risk.mean()
    return avg_expected_risk

def main():
    """ 
    Creates toy data and risks, and then runs min risk training.
    Summary budget is 3 sentences. Lengths of the inputs are sampled
    uniformly from (1 + budget, 5 * budget). 3 sentences will have risk of -1
    while the remaining sentences will have risk 1. 
    Here we represent each sentence as a logit parameter of a bernoulli 
    bernoulli distribution (i.e. the outputs of the bi-rnn model in our 
    actual non-toy code). Min risk training should adjust the logits
    so that the logits corresponding to -1 risk sentences are > 0, 
    while all others are < 0.
    The best expected risk we can achieve here is -3.
    """

    num_inputs = 2
    num_samples = 5
    budget = 3
    lengths = Variable(
        torch.LongTensor(num_inputs).random_(1 + budget, 5 * budget))
    seq_size = lengths.data.max()
    mask = make_mask(lengths)
    logits = nn.Parameter(torch.FloatTensor(num_inputs, seq_size).normal_())
    risk_table = sample_risk(num_inputs, lengths, budget) 
    print("True sentencewise risks:")
    print(risk_table)
    print("")

    def get_risk(samples):
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
                summary_size = 0
                for step in range(seq_size):
                    if samples.data[b, s, step] == 1:
                        sample_risk[b, s] += risk_table[b, step]
                        summary_size += 1
                    if summary_size > budget:
                        break
        return Variable(sample_risk)                

    optim = torch.optim.Adam([logits], lr=.1)
    max_steps = 3000

    avg_expected_risks = []

    for step in range(1, max_steps + 1):
        optim.zero_grad()
        avg_expected_risk = forward(
            logits, mask, lengths, num_samples, get_risk)
        avg_expected_risks.append(avg_expected_risk.data[0])
        sys.stdout.write("{}/{} E[R]={:4.3f}\r".format(
            step, max_steps, np.mean(avg_expected_risks)))
        sys.stdout.flush()
        avg_expected_risk.backward()
        optim.step()
    print("")
    print("Min Risk Prediction:")
    print(logits.gt(0).masked_fill(mask, 0))

    print("Check to see that the min risk prediction is selecting sentences" \
        " with -1 risk.") 

if __name__ == "__main__":
    main()
