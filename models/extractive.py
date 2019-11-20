import torch
import torch.nn as nn
import math
from transformers import BertModel, BertConfig

import pdb

class Bert(nn.Module):
    def __init__(self, finetune):
        super(Bert, self).__init__()
        # model = 'bert-base-uncased', 'bert-large-uncased'
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.finetune = finetune

    def expand_positional_embedding(self, new_position_embeddings_size):
        #  note that BERT only expects a sequence of maximum length = 512
        # (position_embeddings): Embedding(512, 768)
        # we just copy the last position embedding vector for the tokens beyond 512
        # the idea is from https://github.com/nlpyang/PreSumm/blob/master/src/models/model_builder.py line 150
        new_pos_embeddings = nn.Embedding(new_position_embeddings_size, self.model.config.hidden_size)
        new_pos_embeddings.weight.data[:512] = self.model.embeddings.position_embeddings.weight.data
        new_pos_embeddings.weight.data[512:] = self.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(new_position_embeddings_size-512,1)
        self.model.embeddings.position_embeddings = new_pos_embeddings
        self.model.config.max_position_embeddings = new_position_embeddings_size # maybe change this as well?

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids=None):
        """
        input_ids: torch.LongTensor of shape (batch_size, sequence_length)
        attention_mask: (optional) torch.FloatTensor of shape (batch_size, sequence_length) - 1 NOT MASKED, 0 MASKED (to avoid attention on padding token indices)
        token_type_ids: torch.LongTensor of shape (batch_size, sequence_length)
        position_ids: (optional) torch.LongTensor of shape (batch_size, sequence_length) - it will automatically creates 0,1,..,last_pos
        """
        if self.finetune:
            # change the weights
            last_hidden_state, _ = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids, position_ids=position_ids)
        else:
            # no changes to the weights
            self.eval()
            with torch.no_grad():
                output, _ = self.model(input_ids=input_ids, attention_mask=mask,
                                       token_type_ids=token_type_ids, position_ids=position_ids)

        return last_hidden_state # (batch_size, sequence_length, hidden_size)


class ExtractiveTransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, max_len):
        super(ExtractiveTransformerEncoder, self).__init__()

        from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

        d_model = hidden_size # the number of expected features in the input
        nhead = 8 # the number of heads in the multiheadattention models
        dim_feedforward = 2048
        dropout = 0.1

        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        transformer_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers, norm=None)

    def forward(self, x, mask=None, key_padding_mask=None):
        """
        x = BERT output of size (batch_size, sequence_length, hidden_size)
        mask = [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.

        key_padding_mask = [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.
        """
        x = self.positional_encoder(x)
        output = self.transformer_encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)
        return output

class SentClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(SentClassifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x.size() => (batch_size, max_num_sentences, hidden_size)
        output   => (batch_size, max_num_sentences, 1)
        note that the transformer layer (internally) maps from bert_represention -> ff -> bert_represention
        """

        z = self.linear1(x)
        sent_scores = self.sigmoid(z)

        return sent_scores

class ExtractiveSummariser(nn.Module):
    def __init__(self, args, device):
        super(ExtractiveSummariser, self).__init__()
        self.args = args
        self.device = device

        self.bert = Bert(finetune=True)
        if args['max_pos_embed'] == 512:
            pass
        elif args['max_pos_embed'] == 1024:
            self.bert.expand_positional_embedding(self.bert.model.config.max_position_embeddings * 2) # 512 -> 1024
        else:
            raise ValueError("max_pos_embed = 512 or 1024")

        hidden_size = self.bert.model.config.hidden_size # 768

        self.ext_transformer = ExtractiveTransformerEncoder(hidden_size, num_layers=2, max_len=self.args['max_num_sentences'])
        self.sent_classifer = SentClassifier(hidden_size)

        # Initialise the Transformer & Classifier
        # zero out the bias term
        # print("dear zero initialisation, name = {}".format(name))
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.ext_transformer.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if name[-4:] == 'bias': nn.init.zeros_(p)

        for name, p in self.sent_classifer.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if name[-4:] == 'bias': nn.init.zeros_(p)

        # move all weights of all the layers to GPU (if device = cuda)
        self.to(device)

    def forward(self, inputs, key_padding_mask):
        """
        # src = ...
        # cls_pos = postions of the CLS tokens (begining of sentences)
        """
        input_ids, attention_mask, token_type_ids, cls_pos = inputs

        # Size....
        # bert_output => (batch_size, sequence_length, hidden_size)
        # sent_vecs   => (batch_size,  num_sentences,  hidden_size)
        # sent_scores => (batch_size,  num_sentences)

        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        # N = batch_size
        N = bert_output.shape[0]
        sent_vecs = [None for x in range(N)]

        for i, doc in enumerate(bert_output):
            sent_vecs[i] = doc[cls_pos[i],:]

        # pad the first sentence manually to max_sent_length
        # but torch.nn.utils.rnn.pad_sequence will look for sequences of longest length
        for i in range(N):
            if sent_vecs[i].shape[0] > self.args['max_num_sentences']:
                sent_vecs[i] = sent_vecs[i][:self.args['max_num_sentences'],:]

        pad_size = self.args['max_num_sentences'] - sent_vecs[0].shape[0]

        # pad_size must be positive!
        if pad_size <= 0:
            sent_vecs[0] = sent_vecs[0][:self.args['max_num_sentences'],:]
        else:
            sent_vecs[0] = torch.cat((sent_vecs[0], torch.zeros(pad_size, sent_vecs[0].shape[-1]).to(self.device)))

        sent_vecs_padded = torch.nn.utils.rnn.pad_sequence(sent_vecs, batch_first=True)

        # key_padding_mask => (batch_size, num_sentences)
        # key_padding_mask = [None for x in range(N)]
        # for i in range(N):
        #     slen = len(cls_pos[i])
        #     if slen <= self.args['max_num_sentences']:
        #         key_padding_mask[i] = [False]*slen + [True]*(self.args['max_num_sentences']-slen)
        #     else:
        #         key_padding_mask[i] = [False]*self.args['max_num_sentences']
        # key_padding_mask = torch.tensor(key_padding_mask, dtype=KEYPADMASK_DTYPE).to(self.device)

        # input to transformer => (sequence_length, batch_size, hidden_size)
        # tranpose before feedng it to the Transformer then tranpose back ---> maybe there is a better way??
        sent_vecs_padded = torch.transpose(sent_vecs_padded, 0, 1)
        sent_vecs_trans = self.ext_transformer(sent_vecs_padded, key_padding_mask=key_padding_mask)
        sent_vecs_trans = torch.transpose(sent_vecs_trans, 0, 1)

        sent_scores = self.sent_classifer(sent_vecs_trans).squeeze(-1)

        return sent_scores

# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------- Helper Classes ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------ #
class PositionalEncoding(nn.Module):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # it should automatically broadcast the 'batch_size' dimension. --- which is the second dim!!
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
