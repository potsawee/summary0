import torch
import pickle
import pdb
import os
from tqdm import tqdm
from nltk import tokenize

from transformers import BertTokenizer

CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
EOS_TOKENS = ['.', '?', '!']


EMPTY_ARTICLE_TEST_IDS = [4309]
EMPTY_ARTICLE_VAL_IDS = [1174, 7812]

class ProcessedDocument(object):
    def __init__(self, encoded_article, attention_mask, token_type_ids, cls_pos):
        self.encoded_article = encoded_article
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_pos = cls_pos

class ProcessedSummary(object):
    def __init__(self, encoded_abstract, length):
        self.encoded_abstract = encoded_abstract
        self.length = length

class CNNDMloader(object):
    def __init__(self, max_sent_length, max_summary_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_sent_length = max_sent_length
        self.max_summary_length = max_summary_length

    def process_data_article(self, filepaths, data_type):
        if data_type == 'train':
            documents = load_data_pickle(filepaths['train'])
        elif data_type == 'val':
            documents = load_data_pickle(filepaths['val'])
        elif data_type == 'test':
            documents = load_data_pickle(filepaths['test'])
        else:
            raise Exception("train/val/test only")

        processed_documents = [None] * len(documents)

        for idx, document in tqdm(enumerate(documents)):
            article = document[0].decode('utf-8')
            abstract = document[1].decode('utf-8')

            # # PAD [CLS] at the beginning
            # tokenized_article = [CLS_TOKEN]
            #
            # for tok in self.tokenizer.tokenize(article):
            #     if tok in EOS_TOKENS:
            #         # ADD [SEP] and [CLS] between sentences
            #         tokenized_article.append(tok)
            #         tokenized_article.append(SEP_TOKEN)
            #         tokenized_article.append(CLS_TOKEN)
            #     else:
            #         tokenized_article.append(tok)
            #
            # # Remove the last [CLS]
            # tokenized_article = tokenized_article[:-1]

            # Use NLTK instead of my own way!!
            sentences = tokenize.sent_tokenize(article)
            tokenized_article = []
            for sent in sentences:
                # ADD [CLS] at the beginning & [SEP] at the end
                words = [CLS_TOKEN] + self.tokenizer.tokenize(sent) + [SEP_TOKEN]
                tokenized_article += words


            # If the sequence is too long, truncate it! currently should be 1024 or 512
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

        with open("lib/model_data/{}-{}.dat.nltk.pk.bin".format(data_type, self.max_sent_length), "wb") as f:
            pickle.dump(processed_documents, f)

    def process_data_abstract(self, filepaths, data_type):
        # this method mimics process_data_article
        if data_type == 'train':
            documents = load_data_pickle(filepaths['train'])
        elif data_type == 'val':
            documents = load_data_pickle(filepaths['val'])
        elif data_type == 'test':
            documents = load_data_pickle(filepaths['test'])
        else:
            raise Exception("train/val/test only")

        processed_summaries = [None] * len(documents)

        for idx, document in tqdm(enumerate(documents)):
            abstract = document[1].decode('utf-8')
            # note that the abstract is in this format:
            # <s> sent1 </s> <s> sent2 </s> <s> sent3 </s>
            # it will be processed into:
            # [CLS] sent1 [SEP] sent2 [SEP] sent3 [MASK]
            abstract = abstract.replace('</s> <s>', SEP_TOKEN)
            abstract = abstract.replace('<s>', CLS_TOKEN).replace('</s>', MASK_TOKEN)

            tokenized_abstract = self.tokenizer.tokenize(abstract)

            # If the sequence is too long, truncate it! currently should be 96
            if len(tokenized_abstract) > self.max_summary_length:
                tokenized_abstract = tokenized_abstract[:self.max_summary_length]

            encoded_abstract = [self.tokenizer.convert_tokens_to_ids(MASK_TOKEN)] * self.max_summary_length

            for i, tok in enumerate(tokenized_abstract):
                encoded_abstract[i] = self.tokenizer.convert_tokens_to_ids(tok)

            length = len(tokenized_abstract)
            processed_summaries[idx] = ProcessedSummary(encoded_abstract, length)

        with open("lib/model_data/abstract.{}-{}.pk.bin".format(data_type, self.max_summary_length), "wb") as f:
            pickle.dump(processed_summaries, f)

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

def load_extractive_labels(data_type, max_num_sentences):
    if data_type == "test":
        data_dir = "/home/alta/summary/pm574/oracle/extractive_idx-v2/test/"
        num_data = 11490
    elif data_type == "val":
        data_dir = "/home/alta/summary/pm574/oracle/extractive_idx-v2/val/"
        num_data = 13368
    elif data_type == "train":
        data_dir = "/home/alta/summary/pm574/oracle/extractive_idx-v2/train/"
        num_data = 287227
    else:
        raise Exception("Please choose train/val/test")
        print("loading extractive labels from:", data_dir)

    target_positions = [None for x in range(num_data)]

    for idx in tqdm(range(num_data)):
        filepath = data_dir + "idx.{}.txt".format(idx)
        # for train/ => the empty files are not even created!
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                line = f.readline()
            try:
                labels = [int(x) for x in line.split(',') if int(x) < max_num_sentences ]
            except ValueError:
                # empty line in the index file
                # test id: 4309
                if data_type == "test" and idx in EMPTY_ARTICLE_TEST_IDS:
                    labels = []
                elif data_type == "val" and idx in EMPTY_ARTICLE_VAL_IDS:
                    labels = []
                elif data_type == "train":
                    if line == "":
                    	labels = []
                    else:
                        raise Exception("data error #1")
                else:
                    print(data_type)
                    print(idx)
                    raise Exception("data error #2")
        else:
            labels = []

        target_positions[idx] = sorted(labels)

    return target_positions

def exp():
    filepaths = {}
    filepaths['train'] = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/train.pk.bin'
    filepaths['val'] = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/val.pk.bin'
    filepaths['test'] = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/test.pk.bin'
    cnndmloader = CNNDMloader(max_sent_length=512, max_summary_length=96)
    # cnndmloader.process_data_article(filepaths, 'train')
    cnndmloader.process_data_abstract(filepaths, 'train')
    return


def cleanup_work():
    # there are 114 files in the training data that have missing articles!
    # this work aims to remove them!
    path_data        = "lib/model_data/train-512.dat.nltk.pk.bin"
    path_target      = "lib/model_data/target.train-32.pk.bin"
    path_abstract    = "lib/model_data/abstract.train-96.pk.bin"
    with open(path_data, "rb") as f: data = pickle.load(f)
    with open(path_target, "rb") as f: target_pos = pickle.load(f)
    with open(path_abstract, "rb") as f: summaries = pickle.load(f)

    assert len(data) == len(target_pos), "len(data) != len(target_pos)"
    assert len(data) == len(summaries), "len(data) != len(summaries)"

    num_data = len(data)
    count    = 0
    clean_data = []
    clean_target_pos = []
    clean_summaries = []
    for i, doc in enumerate(data):
        # encoded_articles[i]   = doc.encoded_article
        # attention_masks[i]    = doc.attention_mask
        # token_type_ids_arr[i] = doc.token_type_ids
        # cls_pos_arr[i]        = doc.cls_pos
        # target                = target_pos[i]
        if len(doc.cls_pos) == 0:
            count += 1
        else:
            clean_data.append(doc)
            clean_target_pos.append(target_pos[i])
            clean_summaries.append(summaries[i])
    print("remove {} files out of {} in original".format(count, num_data))

    # with open("lib/model_data/trainx-512.dat.nltk.pk.bin", "wb") as f:
    #     pickle.dump(clean_data, f)
    # with open("lib/model_data/target.trainx-32.pk.bin", "wb") as f:
    #     pickle.dump(clean_target_pos, f)
    with open("lib/model_data/abstract.trainx-96.pk.bin", "wb") as f:
        pickle.dump(clean_summaries, f)
    print("cleanup done & saved")

if __name__ == "__main__":
    # exp()
    # x = load_extractive_labels('test')
    # target_pos = load_extractive_labels('train', max_num_sentences=32)
    # pdb.set_trace()
    # print("data2 exp done!")
    cleanup_work()
