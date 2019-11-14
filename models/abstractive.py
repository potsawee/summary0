import torch
import torch.nn as nn
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

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_embed = self.decoder_embedding(tgt)

        tgt_embed = self.positional_encoder(tgt_embed)

        # inputs into the Transformer have batch_size in the 2nd dim
        tgt_embed = torch.transpose(tgt_embed, 0, 1)
        memory    = torch.transpose(memory, 0, 1)
        output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=None,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=None)

        return torch.transpose(output, 0, 1)


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

        # move all weights of all the layers to GPU (if device = cuda)
        self.to(device)

    def forward(self, abs_batch):
        # check that source and target match
        # enc_inputs:  inputs to the BERT model
        # enc_targets: extractive target labels
        # enc_ms:      mask & length for computing extractive loss
        # tgt_ids:     input to the decoder
        # tgt_key_padding_mask: mask for the decoder
        enc_inputs, enc_targets, enc_ms    = abs_batch['encoder']
        tgt_ids, tgt_key_padding_mask = abs_batch['decoder']
        if enc_inputs[0].size(0) != tgt_ids.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        # enc_output => (batch_size, max_pos_embed,      hidden_size)
        # dec_output => (batch_size, max_summary_length, hidden_size)
        enc_output = self.encoder(enc_inputs) # memory
        dec_output = self.decoder(tgt_ids, enc_output, tgt_mask=None,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
