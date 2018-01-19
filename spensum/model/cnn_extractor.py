import ntp
import torch
import torch.nn as nn
import torch.nn.functional as F
from spensum.functional import sequence_dropout


class CNNExtractor(nn.Module):
    def __init__(self, embedding_size, filter_widths=[1,3],
                 num_filters=100):
        
        super(CNNExtractor, self).__init__()

        self.cnn_ = ntp.modules.EncoderCNN(
            embedding_size, filter_widths, num_filters, dropout=.05)
        
        self.predictor_module_ = ntp.modules.MultiLayerPerceptron(
            embedding_size + num_filters * len(filter_widths), 1, 
            #hidden_sizes=[100],
            #hidden_layer_activations="relu",
           # hidden_layer_dropout=.05,
            output_activation="sigmoid") 
        #self.input_module_ = input_module
        #self.encoder_module_ = encoder_module
        #self.predictor_module_ = predictor_module
        #self.pad_value_ = pad_value


    def forward(self, inputs, mask=None):

        input_embeddings = inputs.embedding #.transpose(1, 0)
        batch_size = input_embeddings.size(0)
        sequence_size = input_embeddings.size(1)
        mask = input_embeddings.eq(-1)

        input_embeddings = sequence_dropout(
            input_embeddings, p=.25, training=self.training,
            batch_first=True)

        input_embeddings = input_embeddings.masked_fill(mask, 0)

        feature_maps = self.cnn.encoder_state_output(
            input_embeddings)
        feature_maps = feature_maps.view(batch_size, 1, -1).repeat(
            1, sequence_size, 1)

        mlp_input = torch.cat([input_embeddings, feature_maps], 2)
        mlp_input_flat = mlp_input.view(batch_size * sequence_size, -1)
        probs_flat = self.predictor_module(mlp_input_flat)
        probs = probs_flat.view(batch_size, sequence_size)
        probs = probs.masked_fill(mask[:,:,0], 0)
        return probs
        
    @property
    def cnn(self):
        return self.cnn_

    @property
    def predictor_module(self):
        return self.predictor_module_

    def extract(self, inputs, metadata, strategy="rank", word_limit=100):
        probs = self.forward(inputs)
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
