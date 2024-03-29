# import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig

from models.extractive import Bert, PositionalEncoding

import pdb

class AbsEncoder(nn.Module):
    def __init__(self, finetune):
        super(AbsEncoder, self).__init__()
        self.bert = Bert(finetune=finetune)

    def forward(self, inputs):
        # bert_output => (batch_size, sequence_length, hidden_size)
        input_ids, attention_mask, token_type_ids, cls_pos = inputs
        # at the moment cls_pos is not used for the abstractive system!
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        return bert_output

class AbsDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, max_len):
        super(AbsDecoder, self).__init__()

        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)

        from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer

        d_model = hidden_size # the number of expected features in the input
        nhead = 8 # the number of heads in the multiheadattention models
        dim_feedforward = 2048
        dropout = 0.1

        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        transformer_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(transformer_decoder_layer, num_layers, norm=None)

        # Linear & Softmax Layers
        self.linear_decoder = nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=True)
        self.logsoftmax_decoder = nn.LogSoftmax(dim=-1)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, logsoftmax=True):
        # tgt                     => [batch_size, tgt_length]
        # memory                  => [batch_size, memory_length, hidden_size]
        # tgt_mask                => [tgt_length, tgt_length]
        # tgt_key_padding_mask    => [batch_size, tgt_length]
        # memory_key_padding_mask => [batch_size, memory_length] # memory_length = max_pos_embed
        tgt_embed = self.decoder_embedding(tgt)

        # inputs into the Transformer have batch_size in the 2nd dim
        tgt_embed = torch.transpose(tgt_embed, 0, 1)
        memory    = torch.transpose(memory, 0, 1)

        tgt_embed = self.positional_encoder(tgt_embed)

        # tgt_mask                => ensure no information from future context (self-attention layer)
        # tgt_key_padding_mask    => ensure no information from padding in decoder (self-attention layer)
        # memory_key_padding_mask => ensure no information from padding in encoder (enc-dec attention)
        output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=None,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)

        output = self.linear_decoder(torch.transpose(output, 0, 1))

        if logsoftmax:
            output = self.logsoftmax_decoder(output)

        return output


class AbstractiveSummariser(nn.Module):
    def __init__(self, args, device):
        super(AbstractiveSummariser, self).__init__()
        self.args = args
        self.device = device
        self.encoder = AbsEncoder(finetune=True)

        hidden_size = self.encoder.bert.model.config.hidden_size # 768
        vocab_size  = self.encoder.bert.model.config.vocab_size  # 30522

        self.decoder = AbsDecoder(vocab_size, hidden_size, num_layers=6,
                                 max_len=self.args['max_summary_length'])

        self.tgt_mask = self._create_tgt_mask(args['max_summary_length'], device)

        # Initialise the Transformer (decoder)
        for name, p in self.decoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                if name[-4:] == 'bias': nn.init.zeros_(p)

        # move all weights of all the layers to GPU (if device = cuda)
        self.to(device)

    def forward(self, abs_batch):
        # enc_inputs:  inputs to the BERT model
        # enc_targets: extractive target labels
        # tgt_ids:     input to the decoder
        # tgt_key_padding_mask: mask for the decoder
        enc_inputs, enc_targets, _, _    = abs_batch['encoder']
        memory_key_padding_mask          = abs_batch['memory']
        tgt_ids, tgt_key_padding_mask, _ = abs_batch['decoder']

        if enc_inputs[0].size(0) != tgt_ids.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        # enc_output => (batch_size, max_pos_embed,      hidden_size)
        # dec_output => (batch_size, max_summary_length, hidden_size)

        enc_output = self.encoder(enc_inputs) # memory
        dec_output = self.decoder(tgt_ids, enc_output,
                                  tgt_mask=self.tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        return dec_output

    def decode_beamsearch(self, enc_inputs, memory_key_padding_mask, decode_dict):
        """
        this method is meant to be used at inference time
            encoder_inputs          = input_dict into the encoder
            memory_key_padding_mask = mask for the output of the encoder
            decode_dict:
                - k                = beamwidth for beamsearch
                - batch_size       = batch_size
                - time_step        = max_summary_length
                - vocab_size       = 30522 for BERT
                - device           = cpu or cuda
                - start_token_id   = ID of the start token
                - stop_token_id    = ID of the stop token
                - keypadmask_dtype = torch.bool
        """
        k                = decode_dict['k']
        batch_size       = decode_dict['batch_size']
        time_step        = decode_dict['time_step']
        vocab_size       = decode_dict['vocab_size']
        device           = decode_dict['device']
        start_token_id   = decode_dict['start_token_id']
        stop_token_id    = decode_dict['stop_token_id']
        alpha            = decode_dict['alpha']
        length_offset    = decode_dict['length_offset']
        keypadmask_dtype = decode_dict['keypadmask_dtype']

        # create beam array & scores
        beams       = [None for _ in range(k)]
        beam_scores = np.zeros((batch_size, k))

        # we should only feed through the encoder just once!!
        enc_output = self.encoder(enc_inputs) # memory

        # we run the decoder time_step times (auto-regressive)
        tgt_ids = torch.zeros((batch_size, time_step), dtype=torch.int64).to(device)
        tgt_ids[:,0] = start_token_id
        for i in range(k):
            beams[i] = tgt_ids

        for t in range(time_step-1):
            # tgt_key_padding_mask
            row_padding_mask = [False]*(t+1) + [True]*(time_step-t-1)
            padding_mask     = [row_padding_mask for _ in range(batch_size)]
            tgt_key_padding_mask = torch.tensor(padding_mask, dtype=keypadmask_dtype).to(device)

            decoder_output_t_array = torch.zeros((batch_size, k*vocab_size))

            for i, beam in enumerate(beams):
                decoder_output = self.decoder(beam, enc_output,
                                          tgt_mask=self.tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask,
                                          logsoftmax=False)

                # check if there is STOP_TOKEN emitted in the previous time step already
                # i.e. if the input at this time step is STOP_TOKEN
                for n_idx in range(batch_size):
                    if beam[n_idx][t] == stop_token_id: # already stop
                        decoder_output[n_idx, t, :] = float('-inf')
                        decoder_output[n_idx, t, stop_token_id] = 0.0 # to ensure STOP_TOKEN will be picked again!

                    else: # need to update scores --- length norm
                        beam_scores[n_idx,i] *= (t-1+length_offset)**alpha
                        beam_scores[n_idx,i] /= (t+length_offset)**alpha

                # length_norm = 1/(length)^alpha ... alpha = 0.7
                decoder_output_t_array[:,i*vocab_size:(i+1)*vocab_size] = decoder_output[:,t,:]/(t+length_offset)**alpha

                # add previous beam score bias
                for n_idx in range(batch_size):
                    decoder_output_t_array[n_idx,i*vocab_size:(i+1)*vocab_size] += beam_scores[n_idx,i]

                if t == 0: break # only fill once for the first time step

            # scores, indice => (batch_size, k)
            scores, indices = torch.topk(decoder_output_t_array, k=k, dim=-1)
            new_beams = [torch.zeros((batch_size, time_step), dtype=torch.int64).to(device) for _ in range(k)]
            for r_idx, row in enumerate(indices):
                for c_idx, node in enumerate(row):
                    vocab_idx = node % vocab_size
                    beam_idx  = int(node / vocab_size)

                    new_beams[c_idx][r_idx,:t+1] = beams[beam_idx][r_idx,:t+1]
                    new_beams[c_idx][r_idx,t+1]  = vocab_idx

            beam_scores = scores.cpu().numpy()
            beams = new_beams

        #     if (t % 10) == 0:
        #         print("{}=".format(t), end="")
        #         sys.stdout.flush()
        # print("{}=#".format(t))

        summaries_id = [None for _ in range(batch_size)]
        for j in range(batch_size): summaries_id[j] = beams[0][j].cpu().numpy()

        return summaries_id


    def _create_tgt_mask(self, tgt_max_length, device):
        # tgt_mask to ensure future information is used
        tgt_mask_shape = (tgt_max_length, tgt_max_length)
        x = np.triu(np.ones(tgt_mask_shape), k=1).astype('float')
        x[x == 1] = float('-inf')
        tgt_mask = torch.from_numpy(x).type(torch.float32).to(device)
        return tgt_mask
