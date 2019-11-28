DATA_DIR   = '/home/alta/summary/pm574/summariser0/lib/ami_goochen/'
CLS_TOKEN  = "[CLS]"
SEP_TOKEN  = "[SEP]"
MASK_TOKEN = "[MASK]"

DA_STR2ID = {'assess': 0, 'stall': 1, 'suggest': 2, 'inform': 3,
         'offer': 4, 'other': 5, 'fragment': 6, 'backchannel': 7,
         'be-positive': 8, 'elicit-inform': 9, 'elicit-assessment': 10,
         'elicit-offer-or-suggestion': 11, 'comment-about-understanding': 12,
         'be-negative': 13, 'elicit-comment-understanding': 14}

DA_ID2STR = dict((v,k) for k,v in DA_STR2ID.items())

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

def load_data(data_type, max_sent_length, max_summary_length):
    data_dict = read_data(DATA_DIR+'{}/'.format(data_type))
    num_data  = len(data_dict['in'])

    # ------------------------------------- INPUT ------------------------------------- #
    processed_documents = [None] * num_data
    for idx in range(num_data):
        utterances = data_dict['in'][idx].replace(".","").split("<eos>")[:-1]
        tokenized_input = []
        for utterance in utterances:
            words = [CLS_TOKEN] + bert_tokenizer.tokenize(utterance) + [SEP_TOKEN]
            tokenized_input += words
        if len(tokenized_input) > max_sent_length:
            tokenized_input = tokenized_input[:max_sent_length]
            print("ID: {} is too long".format(idx))

        encoded_input = [bert_tokenizer.convert_tokens_to_ids('[UNK]')] * max_sent_length
        token_type_ids = [0] * max_sent_length # 0 and 1 alternate
        cls_pos = []
        type_id = 0

        for i, tok in enumerate(tokenized_input):
            encoded_input[i] = bert_tokenizer.convert_tokens_to_ids(tok)
            token_type_ids[i] = type_id
            if tok == CLS_TOKEN:
                cls_pos.append(i)
            if tok == SEP_TOKEN:
                type_id = (type_id + 1) % 2

        # len(tokenized_article) is at most self.max_sent_length
        attention_mask = ([1]*len(tokenized_input) + [0]*(max_sent_length-len(tokenized_input)))

        processed_documents[idx] = ProcessedDocument(encoded_input, attention_mask, token_type_ids, cls_pos)

    # ----------------------------------- SUMMARY ----------------------------------- #
    processed_summaries = [None] * num_data
    for idx in range(num_data):
        summary  = data_dict['sum'][idx]
        # note that the abstract is in this format:
        # <s> sent1 </s> <s> sent2 </s> <s> sent3 </s>
        # it will be processed into:
        # [CLS] sent1 [SEP] sent2 [SEP] sent3 [MASK]
        # but there is only one sentence in this AMI data
        summary  = "{} {} {}".format(CLS_TOKEN, summary, MASK_TOKEN)
        tokenized_sum = bert_tokenizer.tokenize(summary)

        # If the sequence is too long, truncate it! currently should be 96
        if len(tokenized_sum) > max_summary_length:
            tokenized_sum = tokenized_sum[:max_summary_length]

        encoded_sum = [bert_tokenizer.convert_tokens_to_ids(MASK_TOKEN)] * max_summary_length

        for i, tok in enumerate(tokenized_sum):
            encoded_sum[i] = bert_tokenizer.convert_tokens_to_ids(tok)

        length = len(tokenized_sum)
        processed_summaries[idx] = ProcessedSummary(encoded_sum, length)

    # --------------------------------- Dialogue Acts --------------------------------- #
    dialogueacts = [None] * num_data
    for idx in range(num_data):
        acts = data_dict['da'][idx].split()
        dialogueacts[idx] = [DA_STR2ID[act] for act in acts]


    return {'in': processed_documents, 'sum': processed_summaries, 'da': dialogueacts}

def read_data(data_dir):
    # dialogue act
    da = data_dir + 'da'
    with open(da, 'r') as f:
        dialog_acts = f.readlines()
        dialog_acts = [d.strip() for d in dialog_acts]
     # summary
    sum = data_dir + 'sum'
    with open(sum, 'r') as f:
        summaries = f.readlines()
        summaries = [s.strip() for s in summaries]
        # dialog act
    inp = data_dir + 'in'
    with open(inp, 'r') as f:
        inputs = f.readlines()
        inputs = [i.strip() for i in inputs]

    assert len(dialog_acts) == len(inputs)
    assert len(dialog_acts) == len(summaries)

    return {'da': dialog_acts, 'in': inputs, 'sum': summaries}
