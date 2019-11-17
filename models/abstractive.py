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

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
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


    def _create_tgt_mask(self, tgt_max_length, device):
        # tgt_mask to ensure future information is used
        tgt_mask_shape = (tgt_max_length, tgt_max_length)
        x = np.triu(np.ones(tgt_mask_shape), k=1).astype('float')
        x[x == 1] = float('-inf')
        tgt_mask = torch.from_numpy(x).type(torch.float32).to(device)
        return tgt_mask
