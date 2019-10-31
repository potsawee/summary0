import torch
import pickle
import pdb
from tqdm import tqdm

from transformers import BertTokenizer

CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
EOS_TOKENS = ['.', '?', '!']

class ProcessedDocument(object):
    def __init__(self, encoded_article, attention_mask, token_type_ids, cls_pos):
        self.encoded_article = encoded_article
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_pos = cls_pos

class CNNDMloader(object):
    def __init__(self, max_sent_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_sent_length = max_sent_length

    def from_pickle(self, filepaths):
        # note that these documents are in 'bytes'
        # self.train_files = load_data_pickle(filepaths['train'])
        # self.val_files = load_data_pickle(filepaths['val'])
        self.test_files = load_data_pickle(filepaths['test'])

    def process_data(self):
        # input_ids, attention_mask, token_type_ids, cls_pos = inputs

        processed_documents = [None] * len(self.test_files)

        for idx, document in tqdm(enumerate(self.test_files)):
            article = document[0].decode('utf-8')
            abstract = document[1].decode('utf-8')

            # PAD [CLS] at the beginning
            tokenized_article = [CLS_TOKEN]

            for tok in self.tokenizer.tokenize(article):
                if tok in EOS_TOKENS:
                    # ADD [SEP] and [CLS] between sentences
                    tokenized_article.append(tok)
                    tokenized_article.append(SEP_TOKEN)
                    tokenized_article.append(CLS_TOKEN)
                else:
                    tokenized_article.append(tok)

            # Remove the last [CLS]
            tokenized_article = tokenized_article[:-1]

            # If the sequence is too long, truncate it! currently should be 1024
            if len(tokenized_article) > self.max_sent_length:
                tokenized_article = tokenized_article[:self.max_sent_length]

            encoded_article = [self.tokenizer.convert_tokens_to_ids('[UNK]')] * self.max_sent_length
            token_type_ids = [0] * self.max_sent_length # 0 and 1 alternate
            cls_pos = []
            type_id = 0

            for i, tok in enumerate(tokenized_article):
                encoded_article[i] = self.tokenizer.convert_tokens_to_ids(tok)
                token_type_ids[i] = type_id
                if tok == CLS_TOKEN:
                    cls_pos.append(i)
                if tok == SEP_TOKEN:
                    type_id = (type_id + 1) % 2

            # len(tokenized_article) is at most self.max_sent_length
            attention_mask = ([1]*len(tokenized_article) + [0]*(self.max_sent_length-len(tokenized_article)))

            processed_documents[idx] = ProcessedDocument(encoded_article, attention_mask, token_type_ids, cls_pos)

        pdb.set_trace()
        # with open("test31oct.dat.pk.bin", "wb") as f:
        #     f.dump(processed_documents, f)


def load_data_pickle(path):
    """
    Documents[i] = (article, abstract)
        article  = bytes (to be converted to string by article.decode('utf-8'))
        abstract = bytes (to be converted to string by abstract.decode('utf-8'))
    """
    with open(path, 'rb') as f:
        documents = pickle.load(f, encoding="bytes")
    print("Loaded: {}".format(path))

    return documents


def exp():
    filepaths = {}
    filepaths['train'] = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/train.pk.bin'
    filepaths['val'] = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/val.pk.bin'
    filepaths['test'] = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/test.pk.bin'
    cnndmloader = CNNDMloader(max_sent_length=1024)
    cnndmloader.from_pickle(filepaths)
    cnndmloader.process_data()
    return

if __name__ == "__main__":
    exp()
