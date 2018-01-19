import torch.nn as nn
from ..module import SpenModule


import ntp
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from spensum.functional import sequence_dropout
import math


class EnergyModel2(nn.Module):

    def __init__(self, submodules, mask_value=-1, search_iters=15, lr=.1, 
                 tol=1e-6):
        super(EnergyModel2, self).__init__()

        #for module in submodules:
        #    if not issubclass(module, SpenModule):
        #        raise Exception("submodules must inherit from SpenModule")

        self.mask_value = mask_value
        self.search_iters = search_iters
        self.lr = lr
        self.tol = tol
        self.submodules_ = nn.ModuleList(submodules)

    @property
    def submodules(self):
        return self.submodules_

    @property
    def ready(self):
        return all([m.ready for m in self.submodules])

    @property
    def burn_in_iters(self):
        return max([m.burn_in_iters for m in self.submodules])

    #@property
    #def burn_in_iters(self):
        



#        print("Yuss")
#        exit()
#
#        self.lr = lr
#        self.search_iters = search_iters
#        self.tol = tol
#
#        self.position_params = nn.Parameter(torch.randn(1,1000))
#        
#        self.wc_weight = nn.Parameter(torch.randn(1, 1, 1))
#        self.wc_bias = nn.Parameter(torch.randn(1, 1, 1))
#
#
#
#        self.rnn = ntp.modules.EncoderRNN(
#            embedding_size, embedding_size, rnn_cell=rnn_cell,
#            layers=layers, bidirectional=bidirectional, merge_mode=merge_mode)
#
#        self.rnn2 = ntp.modules.EncoderRNN(
#            embedding_size, embedding_size // 2, rnn_cell=rnn_cell,
#            layers=layers, bidirectional=bidirectional, merge_mode=merge_mode)
#
#
#
#
#
#        ctx_size = embedding_size   
#        if bidirectional and merge_mode == "concat":
#            ctx_size *= 2
#
#        self.salience_module = ntp.modules.MultiLayerPerceptron(
#            ctx_size, 1, output_activation=None)
#
#        self.coverage_module = ntp.modules.MultiLayerPerceptron(
#            1, 1, output_activation=None)


    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):

        features = []
        for module in self.submodules:
            features.append(
                module.compute_features(
                    inputs, inputs_mask=inputs_mask, 
                    targets_mask=targets_mask))
        return features




    def compute_features2(self, inputs):
        emb_mask = inputs.embedding.eq(-1)
        

        input_embeddings_bf = inputs.embedding.masked_fill(emb_mask, 0)
        
        input_sum = input_embeddings_bf.transpose(2,1).sum(2, keepdim=True)

        input_mean = Variable(
            (input_sum / Variable(
                inputs.length.view(-1, 1, 1).data.float())).data)

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
        salience_flat = self.salience_module(context_flat)
        salience = salience_flat.view(sequence_size, batch_size).transpose(1, 0)
        salience = salience.masked_fill(mask, 0)

        context_sequence = self.rnn2.encoder_context(
            input_embeddings, length=input_lengths)
        context_sequence = sequence_dropout(
            context_sequence, p=.25, training=self.training)

        #print(input_mean.size())
        #print(context_sequence.size())
        #print(input_mean.repeat(1,2,1).size())
        coverage_us = context_sequence.transpose(1,0).bmm(
            input_mean).view(batch_size, -1)
        #exit()
        #coverage_us = inputs.embedding.bmm(input_mean).view(batch_size, -1)
        coverage_us = coverage_us / Variable(
            inputs.length.data.float().view(-1, 1))

        coverage_us_flat = coverage_us.view(-1,1)
        coverage_flat = self.coverage_module(coverage_us_flat)
        coverage = coverage_flat.view(batch_size, sequence_size)
        coverage = coverage.masked_fill(mask, 0)

        position = self.position_params[:,:sequence_size].repeat(
            batch_size, 1).masked_fill(mask, 0)

        wc = (self.wc_weight * inputs.word_count + self.wc_bias).view(
            batch_size, sequence_size).masked_fill(mask, 0)
        return salience, coverage, position, wc
 
    def forward(self, inputs, targets, inputs_mask=None, targets_mask=None, 
                precomputed=None):

        if precomputed is None:
            features = self.compute_features(inputs)
        else:
            features = precomputed
        
        self.submodule_energy = []

        for module, feature in zip(self.submodules, features):
            module_energy = module.compute_energy(
                inputs, feature, targets, inputs_mask=inputs_mask,
                targets_mask=targets_mask)
            self.submodule_energy.append(module_energy)

        self.submodule_energy = torch.cat(self.submodule_energy, 1)
        total_energy = self.submodule_energy.sum(1)
        
        return total_energy

        exit()

        salience = torch.sigmoid(salience)

        coverage = torch.sigmoid(coverage)
        position = torch.sigmoid(position)
        wc = torch.sigmoid(wc)
        
        sal_energy_pw = -targets * salience - (1 - targets) * (1 - salience)
        cov_energy_pw = -targets * coverage - (1 - targets) * (1 - coverage)
        pos_energy_pw = -targets * position - (1 - targets) * (1 - position)
        wc_energy_pw = -targets * wc - (1 - targets) * (1 - wc)
        
        sal_energy = sal_energy_pw.sum(1) / Variable(
            inputs.length.data.float())
        cov_energy = cov_energy_pw.sum(1) / Variable(
            inputs.length.data.float())
        pos_energy = pos_energy_pw.sum(1) / Variable(
            inputs.length.data.float())
        wc_energy = wc_energy_pw.sum(1) / Variable(
            inputs.length.data.float())


        
        emb_mask = inputs.embedding.eq(-1)
        sent_embedding = inputs.embedding.masked_fill(emb_mask, 0).data
        #print(F.cosine_similarity(sent_embedding, sent_embedding, 2))
        norms = torch.norm(sent_embedding, 2, 2, keepdim=True)

        #print(norms)
        normalizer = 1 / norms.bmm(norms.transpose(2,1))
        #print(normalizer)
        normalizer = normalizer.masked_fill_(normalizer.eq(float("inf")), 1)
        #print(normalizer)
        K = Variable(sent_embedding.bmm(sent_embedding.transpose(2,1)) \
            * normalizer)

        batch_size = targets.size(0)
        seq_size = targets.size(1)

        K_weighted = K.bmm((1 - targets.unsqueeze(2)))

        cov2 = -targets.unsqueeze(1).bmm(K_weighted).view(-1) / Variable(
            inputs.length.data.float().view(-1))

        l1 = .01 * targets.sum(1)
        self.module_energy = torch.cat(
            [sal_energy.view(-1, 1).data, 
             cov_energy.view(-1, 1).data,
             pos_energy.view(-1, 1).data,
             wc_energy.view(-1, 1).data,
             cov2.view(-1, 1).data,
             l1.view(-1, 1).data], 1)


        energy_total = sal_energy + cov_energy + pos_energy + wc_energy \
                + cov2 + l1




#        up_energy = targets * (salience + coverage)
#        down_energy = (1 - targets) * ((1-salience) + (1- coverage))
#        elem_energy = -(up_energy + down_energy)
#        energy_total = elem_energy.sum(1) / Variable(
#            inputs.length.data.float())

#        self.module_energy

        return energy_total

    def search(self, inputs, inputs_mask=None, targets_mask=None, round=True):

        if targets_mask is None:
            targets_mask = inputs.embedding[:,:,0].eq(self.mask_value)

        features = []
        for feature in self.compute_features(
                inputs, targets_mask=targets_mask):
            features.append(Variable(feature.data))

        targets_log = self.submodules[0].forward_pass(
            inputs, features[0], inputs_mask=inputs_mask, 
            targets_mask=targets_mask)

        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)

        #targets_log = inputs.embedding.data.new().resize_(
        #    batch_size, input_size)
        #targets_log = targets_log.normal_(0, .001)
        targets_log = nn.Parameter(targets_log.data)
        opt = torch.optim.Adam([targets_log], lr=self.lr)

        prev_energy = float("inf")
        for i in range(self.search_iters):
            opt.zero_grad() 
       
            targets = torch.sigmoid(targets_log)
            targets = targets.masked_fill(targets_mask, 0)

            energy = self.forward(
                inputs, targets, targets_mask=targets_mask, 
                precomputed=features)
            avg_energy = energy.mean()
            avg_energy.backward()
            
            opt.step()

            if math.fabs(prev_energy - avg_energy.data[0]) < self.tol:
                break
            prev_energy = avg_energy.data[0]

        targets = torch.sigmoid(targets_log).masked_fill(targets_mask, 0)
        if round:
            return Variable(targets.data.gt(.5).float())
        else:
            return Variable(targets.data)

    def extract(self, inputs, metadata, strategy="rank", word_limit=100):
        targets_mask = inputs.embedding[:,:,0].eq(self.mask_value)
        probs = self.search(inputs, targets_mask=targets_mask, round=False)
        summaries = []
        if strategy == "rank":
            scores, indices = torch.sort(probs, 1, descending=True)
            for b in range(probs.size(0)):
                words = 0
                lines = []
                for i in range(probs.size(1)):
                    idx = indices.data[b][i]
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
