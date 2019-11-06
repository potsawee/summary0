import torch
import torch.nn as nn
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
    def __init__(self, hidden_size, num_layers=1):
        super(ExtractiveTransformerEncoder, self).__init__()

        from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

        d_model = hidden_size # the number of expected features in the input
        nhead = 8 # the number of heads in the multiheadattention models
        dim_feedforward = 2048
        dropout = 0.1

        transformer_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers, norm=None)

    def forward(self, x):
        """
        x = BERT output of size (batch_size, sequence_length, hidden_size)
        """
        output = self.transformer_encoder(x, mask=None)
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


class ExtractiveSummeriser(nn.Module):
    def __init__(self, args, device):
        super(ExtractiveSummeriser, self).__init__()
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

        self.ext_transformer = ExtractiveTransformerEncoder(hidden_size, num_layers=2)
        self.sent_classifer = SentClassifier(hidden_size)

        # Initialise the Transformer & Classifier
        # zero out the bias term
        # print("dear zero initialisation, name = {}".format(name))
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.ext_transformer.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                if name[-4:] == 'bias': p.data.zero_()

        for name, p in self.sent_classifer.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                if name[-4:] == 'bias': p.data.zero_()

        # move all weights of all the layers to GPU (if device = cuda)
        self.to(device)

    def forward(self, inputs):
        """
        # src = ...
        # cls_pos = postions of the CLS tokens (begining of sentences)
        """

        # TODO: clean up this
        input_ids, attention_mask, token_type_ids, cls_pos = inputs

        # Size....
        # bert_output => (batch_size, sequence_length, hidden_size)
        # sent_vecs   => (batch_size,  num_sentences,  hidden_size)
        # sent_scores => (batch_size,  num_sentences)

        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        sent_vecs = [None for x in range(bert_output.shape[0])]
        for i, doc in enumerate(bert_output):
            sent_vecs[i] = doc[cls_pos[i],:]

        # pad the first sentence manually to max_sent_length
        # but torch.nn.utils.rnn.pad_sequence will look for sequences of longest length
        for i in range(len(sent_vecs)):
            if sent_vecs[i].shape[0] > self.args['max_num_sentences']:
                sent_vecs[i] = sent_vecs[i][:self.args['max_num_sentences'],:]

        pad_size = self.args['max_num_sentences'] - sent_vecs[0].shape[0]

        # pad_size must be positive!
        if pad_size <= 0:
            sent_vecs[0] = sent_vecs[0][:self.args['max_num_sentences'],:]
        else:
            sent_vecs[0] = torch.cat((sent_vecs[0], torch.zeros(pad_size, sent_vecs[0].shape[-1]).to(self.device)))

        sent_vecs_padded = torch.nn.utils.rnn.pad_sequence(sent_vecs, batch_first=True)

        sent_vecs_trans = self.ext_transformer(sent_vecs_padded)

        sent_scores = self.sent_classifer(sent_vecs_trans).squeeze(-1)

        return sent_scores
