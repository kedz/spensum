import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spensum.model.sequence_standardizer import SequenceStandardizer
from spensum.functional import sequence_dropout

class RougePredictor(nn.Module):
    def __init__(self, input_size=300, hidden_size=300, dropout=.5,
                 standardize_input=True,
                 relative_positions=4, absolute_positions=50, word_counts=100,
                 tfidfs=12, embedding_size=50):
    
        super(RougePredictor, self).__init__()

        self.relative_positions = relative_positions
        self.absolute_positions = absolute_positions
        self.tfidfs = tfidfs
        self.word_counts = word_counts
        self.rpos_emb = nn.Embedding(
            relative_positions + 1, embedding_size, padding_idx=0)
        self.apos_emb = nn.Embedding(
            absolute_positions + 1, embedding_size, padding_idx=0)
        self.wc_emb = nn.Embedding(
            word_counts + 1, embedding_size, padding_idx=0)
        self.tfidf_emb = nn.Embedding(
            tfidfs + 1, embedding_size, padding_idx=0)
        

        self.dropout = dropout

        self.rnn = nn.GRU(
            input_size, hidden_size, bidirectional=True, num_layers=1)

        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(
            hidden_size * 2 + embedding_size * 4, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        if standardize_input:
            self.standardizer = SequenceStandardizer(input_size)
        else:
            self.standardizer = None

    def forward(self, inputs):
        mask = inputs.sequence[:,:,0:1].ne(-1).float()
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
        hidden_states_cat, _ = nn.utils.rnn.pad_packed_sequence(
                packed_context, batch_first=True)
        hidden_states_cat = sequence_dropout(
            hidden_states_cat, p=self.dropout, training=self.training,
            batch_first=True)


        hidden_states = sequence_dropout(
            F.tanh(self.linear1(hidden_states_cat)), p=self.dropout,
            training=self.training, batch_first=True)

        doc_rep = ((hidden_states * mask).sum(1) \
                / inputs.length.view(-1, 1).float()).view(
                batch_size, 1, -1).repeat(1, seq_size, 1)
        
        abs_pos = self.apos_emb(
            torch.clamp(inputs.absolute_position, 0, self.absolute_positions))
        abs_pos = sequence_dropout(
            abs_pos, p=self.dropout, training=self.training, batch_first=True)
        rel_pos = self.rpos_emb(
            torch.clamp(inputs.relative_position, 0, self.relative_positions))
        rel_pos = sequence_dropout(
            rel_pos, p=self.dropout, training=self.training, batch_first=True)
        wc = self.wc_emb(
            torch.clamp(inputs.word_count, 0, self.word_counts))
        wc = sequence_dropout(
            wc, p=self.dropout, training=self.training, batch_first=True)
        tfidf = self.tfidf_emb(
            torch.clamp(inputs.mean_tfidf, 0, self.tfidfs))
        tfidf = sequence_dropout(
            tfidf, p=self.dropout, training=self.training, batch_first=True)




        pred_input = torch.cat(
            [inputs_sequence, 
             doc_rep, abs_pos, rel_pos, wc, tfidf],
            2)
        
        layer2 = F.relu(self.linear2(pred_input))
        layer2 = sequence_dropout(
            layer2, p=self.dropout, training=self.training, batch_first=True)

        layer3 = F.relu(self.linear3(layer2))
        layer3 = sequence_dropout(
            layer3, p=self.dropout, training=self.training, batch_first=True)
        rouge = F.sigmoid(self.linear4(layer3)).view(batch_size, -1)
        return rouge
