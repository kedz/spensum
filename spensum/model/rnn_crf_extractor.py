import ntp
import torch
import torch.nn as nn
import torch.nn.functional as F
from spensum.functional import sequence_dropout
from ntp.crf.modules import LinearChainDecoder


class RNNCRFExtractor(nn.Module):
    def __init__(self, embedding_size, hidden_size, rnn_cell="lstm", layers=1,
                 bidirectional=True, merge_mode="concat"):
        
        super(RNNCRFExtractor, self).__init__()

        self.rnn_ = ntp.modules.EncoderRNN(
            embedding_size, hidden_size, rnn_cell=rnn_cell, layers=layers,
            bidirectional=bidirectional, merge_mode=merge_mode)
        
        self.predictor_module_ = ntp.modules.MultiLayerPerceptron(
            self.rnn.output_size, 2, output_activation=None) 

        self.decoder_ = LinearChainDecoder(2)

    @property
    def decoder(self):
        return self.decoder_

    def get_features(self, inputs):

        input_embeddings = inputs.embedding.transpose(1, 0)
        sequence_size = input_embeddings.size(0)
        batch_size = input_embeddings.size(1)

        input_embeddings = sequence_dropout(
            input_embeddings, p=.25, training=self.training)

        context_sequence = self.rnn.encoder_context(
            input_embeddings, length=inputs.length)

        context_sequence = sequence_dropout(
            context_sequence, p=.25, training=self.training)

        context_flat = context_sequence.view(sequence_size * batch_size, -1)
        features_flat = self.predictor_module(context_flat)
        features = features_flat.view(sequence_size, batch_size, 2)

        #packed_features = nn.utils.rnn.pack_padded_sequence(
        #        features, inputs.length.data.tolist())

        return features

    def score_state(self, inputs, states, normalized=False):

        features = self.get_features(inputs)
        scores = self.decoder.score_state_sequence(
            features, states.long(), sequence_sizes=inputs.length)
        if normalized:
            log_normalizer = self.decoder.forward_algorithm(
                features, sequence_sizes=inputs.length)
            return scores - log_normalizer

        else:

            return scores

    def predict(self, inputs):

        features = self.get_features(inputs)
        states = self.decoder.viterbi_decode(
            features, sequence_sizes=inputs.length)
        return states
 
    def forward(self, inputs, mask=None):

        input_embeddings = inputs.embedding.transpose(1, 0)
        input_lengths = inputs.length
        sequence_size = input_embeddings.size(0)
        batch_size = input_embeddings.size(1)

        input_embeddings = sequence_dropout(
            input_embeddings, p=.25, training=self.training)
        print(input_embeddings)

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

    def extract(self, inputs, metadata, strategy="rank", word_limit=100):
        
        pred_states = self.predict(inputs)
        print(pred_states)


        exit()

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
