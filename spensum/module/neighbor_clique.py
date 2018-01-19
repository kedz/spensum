from .spen_module import SpenModule
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import ntp
from spensum.functional import sequence_dropout



class NeighborClique(SpenModule):

    def __init__(self, name="NeighborClique", mask_value=-1, burn_in=0):

        super(NeighborClique, self).__init__(
            name=name, mask_value=mask_value, burn_in=burn_in)
        
        self.conv = nn.Conv2d(1, 10, kernel_size=(15,300), padding=(7,0))
        self.ff = nn.Linear(10, 1)
        self.conv2 = nn.Conv2d(1, 10, kernel_size=(15,300), padding=(7,0))
        self.ff2 = nn.Linear(10, 1)
        #self.ff3 = nn.Linear(20, 1)
        #i#self.targets_mlp = ntp.modules.MultiLayerPerceptron(
        #    8, 300 * 4, #hidden_sizes=[100,200],
            #hidden_layer_activations="relu",
            #hidden_layer_dropout=.5,
         #   output_activation="relu")

    def compute_features(self, inputs, inputs_mask=None, targets_mask=None):
        if inputs_mask is None:
            inputs_mask = inputs.embedding.eq(self.mask_value)
        
        inputs_embedding = inputs.embedding.masked_fill(inputs_mask, 0)

        inputs_embedding = sequence_dropout(
            inputs_embedding, p=.5, training=self.training, batch_first=True)

        

        return inputs_embedding

    def forward_pass(self, inputs, features, inputs_mask=None,
                     targets_mask=None):
        raise NotImplementedError()

    def compute_energy(self, inputs, features, targets, inputs_mask=None,
                       targets_mask=None):

        summary_embeddings = targets.unsqueeze(1).unsqueeze(3) * \
                features.unsqueeze(1)
        feature_map = self.conv(summary_embeddings).squeeze(3)
        pointwise_energy = F.relu(F.max_pool1d(
            feature_map, feature_map.size(2)).squeeze(2))
        energy = F.tanh(self.ff(pointwise_energy))


        excluded_embeddings = (1 - targets).unsqueeze(1).unsqueeze(3) * \
                features.unsqueeze(1)
        xfeature_map = self.conv2(excluded_embeddings).squeeze(3)
        xpointwise_energy = F.relu(F.max_pool1d(
            xfeature_map, xfeature_map.size(2)).squeeze(2))
        xenergy = F.tanh(self.ff2(xpointwise_energy))
        
       # cenergy = F.tanh(self.ff3(
       #     torch.cat([pointwise_energy, xpointwise_energy], 1)))

        return torch.cat([energy, xenergy], 1)

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

