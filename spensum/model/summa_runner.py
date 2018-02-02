import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from spensum.functional import sequence_dropout


class SummaRunner(nn.Module):
    def __init__(self, embedding_size=300, hidden_size=200, 
                 relative_positions=4, absolute_positions=50, 
                 position_embedding_size=50, dropout=.5):
    
    #, inputs_size_or_module, hidden_size, 
    #             attention_hidden_size=300, context_dropout=.5, layers=1):
        super(SummaRunner, self).__init__()

        self.dropout = dropout
        self.rnn = nn.GRU(
            embedding_size, hidden_size, bidirectional=True, num_layers=1)
        self.relative_positions = relative_positions
        self.absolute_positions = absolute_positions

        self.rpos_emb = nn.Embedding(
            relative_positions + 1, position_embedding_size, padding_idx=0)
        self.apos_emb = nn.Embedding(
            absolute_positions + 1, position_embedding_size, padding_idx=0)
        self.rpos_layer = nn.Linear(position_embedding_size, 1)
        self.apos_layer = nn.Linear(position_embedding_size, 1)

        self.sent_rep = nn.Linear(hidden_size * 2, hidden_size)
        self.sent_content = nn.Linear(hidden_size, 1)
        self.doc_rep = nn.Linear(hidden_size, hidden_size)
        self.doc_rep2 = nn.Linear(hidden_size, hidden_size)
        self.novelty_layer = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.FloatTensor([0]))

        return
        
    def forward(self, inputs):
        batch_size = inputs.sequence.size(0)
        seq_size = inputs.sequence.size(1)

        abs_pos = self.apos_emb(
            torch.clamp(inputs.absolute_position, 0, self.absolute_positions))
        rel_pos = self.rpos_emb(
            torch.clamp(inputs.relative_position, 0, self.relative_positions))
        
        apos_logit = self.apos_layer(abs_pos).view(batch_size, -1)
        rpos_logit = self.rpos_layer(rel_pos).view(batch_size, -1)
        
        inputs_sequence = sequence_dropout(
            inputs.sequence, p=self.dropout, training=self.training, 
            batch_first=True)
        packed_input = nn.utils.rnn.pack_padded_sequence(
                inputs.sequence, inputs.length.data.tolist(), 
                batch_first=True)
        packed_context, _ = self.rnn(packed_input)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
                packed_context, batch_first=True)

        sentence_states = F.relu(self.sent_rep(hidden_states))
        sentence_states = sequence_dropout(
            sentence_states, p=self.dropout, training=self.training, 
            batch_first=True)


        content_logits = self.sent_content(sentence_states).view(
            batch_size, -1)

        avg_sentence = sentence_states.sum(1).div_(
            inputs.length.view(batch_size, 1).float())
        doc_rep = self.doc_rep2(F.tanh(self.doc_rep(avg_sentence)))
        doc_rep = doc_rep.unsqueeze(2)
        salience_logits = sentence_states.bmm(doc_rep).view(batch_size, -1)
        
        sentence_states = sentence_states.split(1, dim=1)

        logits = []
        summary_rep = Variable(
            sentence_states[0].data.new(sentence_states[0].size()).fill_(0))

        for step in range(seq_size):
            
            squashed_summary = F.tanh(summary_rep.transpose(1, 2))
            novelty_logits = -self.novelty_layer(sentence_states[step]).bmm(
                squashed_summary).view(batch_size)
            
            logits_step = content_logits[:, step] + salience_logits[:, step] \
                + novelty_logits + apos_logit[:, step] + rpos_logit[:, step] \
                + self.bias
            
            prob = F.sigmoid(logits_step)

            summary_rep = summary_rep + sentence_states[step] * prob.view(
                batch_size, 1, 1)
            logits.append(logits_step.view(batch_size, 1))
            
        logits = torch.cat(logits, 1)

        return logits

    def predict(self, inputs, metadata, max_words=100, return_indices=False):
        mask = inputs.sequence.data[:,:,0].eq(-1)
        batch_size = inputs.sequence.size(0)

        probs = self.forward(inputs).data.masked_fill_(mask, float("-inf"))
        _, indices = torch.sort(probs, 1, descending=True)

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
