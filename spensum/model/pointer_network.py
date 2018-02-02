import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from spensum.functional import sequence_dropout

class PointerNetwork(nn.Module):
    def __init__(self, inputs_size_or_module, hidden_size, 
                 attention_hidden_size=300, context_dropout=.5, layers=1):
        super(PointerNetwork, self).__init__()

        self.context_dropout = context_dropout

        if isinstance(inputs_size_or_module, int):
            inputs_size = inputs_size_or_module
            self.input_mod = None
        else:
            self.input_mod = inputs_size_or_module
            inputs_size = self.input_mod.output_size

        self.encoder = nn.GRU(
            inputs_size, hidden_size, bidirectional=False, num_layers=layers)
        self.decoder = nn.GRU(
            inputs_size, hidden_size, bidirectional=False, num_layers=layers)

        self.decoder_start = nn.Parameter(torch.randn(1, 1, inputs_size))
        self.decoder_stop = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Attention Module Parameters
        self.attention_hidden_size = attention_hidden_size
        self.Wenc = nn.Parameter(
            torch.randn(1, attention_hidden_size, hidden_size))
        self.Wdec = nn.Parameter(
            torch.randn(1, attention_hidden_size, hidden_size))
        self.v = nn.Parameter(torch.randn(1, attention_hidden_size, 1))

    def make_decoder_input(self, enc_sequence, dec_sequence):
        batch_size = enc_sequence.size(0)
        enc_seq_size = enc_sequence.size(1)
        dec_seq_size = dec_sequence.size(1)
        dims = enc_sequence.size(2)               
        index = dec_sequence.view(
            batch_size, dec_seq_size, 1).repeat(1, 1, dims)
        index = index.masked_fill(index.eq(-1), enc_sequence.size(1) - 1)
        decoder_input_without_start = enc_sequence.gather(
            1, index).transpose(1, 0)

        decoder_start = self.decoder_start.repeat(1, batch_size, 1)        
        decoder_input = torch.cat(
            [decoder_start, decoder_input_without_start], 0)
        return decoder_input
        
    def forward(self, inputs, targets):
          
        batch_size = inputs.sequence.size(0)
        inputs_mask = self.make_inputs_mask(inputs)
        inputs_mask = inputs_mask.view(
                batch_size, 1, -1).repeat(1, targets.sequence.size(1) + 1, 1)
        for b in range(batch_size):
            for step, idx in enumerate(targets.sequence.data[b]):
                
                inputs_mask[b,step +1:,idx] = 1
        #print(inputs_mask)
        enc_seq_size = inputs.sequence.size(1) + 1
        dec_seq_size = targets.sequence.size(1) + 1
        
        # make inputs 0 mean, unit variance, since tsne coords, 
        # word counts, and salience are all on very different scales
        #input_sequence = self.standardize(inputs.sequence)
        if self.input_mod:
            input_sequence = self.input_mod(inputs)
        else:
            input_sequence = inputs.sequence        

        inputs_packed = nn.utils.rnn.pack_padded_sequence(
                input_sequence, inputs.length.data.tolist(), batch_first=True)
        packed_context, encoder_state = self.encoder(inputs_packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(
                packed_context, batch_first=True)

        decoder_stop = self.decoder_stop.repeat(batch_size, 1, 1)       
        context_states = torch.cat([context, decoder_stop], 1)
        context_states = sequence_dropout(
            context_states, p=self.context_dropout, 
            training=self.training, batch_first=True)
        
        # In training mode copy over target label sequence
        decoder_input = self.make_decoder_input(
            input_sequence, targets.sequence)

        decoder_states = self.decoder(
            decoder_input, encoder_state)[0].transpose(1,0)

        output_logits = self.compute_training_attention_logits(
            context_states, decoder_states)

        output_logits.data.masked_fill_(inputs_mask, float("-inf"))
        
        return output_logits

    def compute_training_attention_logits(self, context_states, 
                                          decoder_states):
        # context_states is a batch size x input sequence size x hidden size
        # decoder_states is a batch size x output sequence size x hidden size
        
        batch_size = context_states.size(0)
        enc_seq_size = context_states.size(1)
        dec_seq_size = decoder_states.size(1)

        WencT = self.Wenc.repeat(batch_size, 1, 1).transpose(2,1)        
        WdecT = self.Wdec.repeat(batch_size, 1, 1).transpose(2,1)
        att_ctx = context_states.bmm(WencT)
        att_dec = decoder_states.bmm(WdecT)
        
        att_flat = att_ctx.view(
            batch_size, 1, -1, self.attention_hidden_size).repeat(
                1, dec_seq_size, 1, 1)
        att_flat.add_(
            att_dec.view(
                batch_size, -1, 1, self.attention_hidden_size).repeat(
                    1, 1, enc_seq_size, 1))
        att_flat = F.tanh(att_flat)
        att_flat = att_flat.view(batch_size, -1, self.attention_hidden_size)
        v = self.v.repeat(batch_size, 1, 1)
        output_logits = att_flat.bmm(v).view(
            batch_size, dec_seq_size, enc_seq_size)
        return output_logits
    
    def make_inputs_mask(self, inputs):
        batch_size = inputs.sequence.size(0)
        max_len = inputs.length.data[0] 
        input_mask = inputs.sequence.data.new().byte()
        input_mask.resize_(batch_size, inputs.length.data[0] + 1).fill_(0)
        for b in range(1, batch_size):
            if inputs.length.data[b] < max_len:
                input_mask[b,inputs.length.data[b]:max_len].fill_(1)            
        return input_mask
    
    def select_inputs(self, input_sequence, index):
        max_val = input_sequence.size(1) - 1
        gather_index = torch.clamp(index.view(-1,1,1), 0, max_val).repeat(1,1,input_sequence.size(2))
        selected = input_sequence.gather(1, gather_index)
        return selected.transpose(1, 0)

    def greedy_predict(self, inputs, max_steps=10):
        
        batch_size = inputs.sequence.size(0)
        inputs_mask = self.make_inputs_mask(inputs)
                
        enc_seq_size = inputs.sequence.size(1) + 1
        #dec_seq_size = targets.sequence.size(1) + 1
        
        if self.input_mod:
            input_sequence = self.input_mod(inputs)
        else:
            input_sequence = inputs.sequence        
             
        inputs_packed = nn.utils.rnn.pack_padded_sequence(
                input_sequence, inputs.length.data.tolist(), batch_first=True)
        packed_context, encoder_state = self.encoder(inputs_packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(
                packed_context, batch_first=True)

        decoder_stop = self.decoder_stop.repeat(batch_size, 1, 1)       
        context = torch.cat([context, decoder_stop], 1)
        WencT = self.Wenc.repeat(batch_size, 1, 1).transpose(2,1)
        WdecT = self.Wdec.repeat(batch_size, 1, 1).transpose(2,1)

        context = context.bmm(WencT)
        
        prev_state = encoder_state
        decoder_input = self.decoder_start.repeat(1, batch_size, 1)
        
        stop_signal = inputs.sequence.size(1)
        is_finished = np.zeros(batch_size)
        output_mask = Variable(inputs.sequence.data.new().byte().resize_(batch_size, max_steps).fill_(0))
        
        #all_decoder_inputs = []
        all_predictions = []
        
        for step in range(max_steps):
            #all_decoder_inputs.append(decoder_input)
            hidden_states, prev_state = self.decoder(decoder_input, prev_state)
            
            # Wenc * c_i + Wdec * h_i  
            att_input = context + hidden_states.transpose(1,0).bmm(WdecT).repeat(1, context.size(1), 1)
           
            # Attention logits
            # v * tanh(Wenc * c_i + Wdec * h_i)
            att_logits = F.tanh(att_input).bmm(self.v.repeat(batch_size, 1, 1)).view(batch_size, -1)
            att_logits.data.masked_fill_(inputs_mask, float("-inf"))
            
            log_probs = F.log_softmax(att_logits, dim=1)
            pred_log_probs, predictions = torch.max(log_probs, 1)

            all_predictions.append(predictions.view(-1, 1))
            
            for b, idx in enumerate(predictions.data):
                if idx != stop_signal:
                    inputs_mask[b, idx] = 1
                else:
                    if is_finished[b] == 0:
                        output_mask.data[b, step:max_steps].fill_(1)
                    is_finished[b] = 1    
                    
            
            decoder_input = self.select_inputs(input_sequence, predictions)
            if np.all(is_finished == 1):
                break
        predictions = torch.cat(all_predictions, 1)
        predictions = predictions.masked_fill(output_mask[:,:predictions.size(1)], -1)
        if predictions.data[:,-1].eq(-1).all():
            predictions = predictions[:,:-1]
        return predictions 
