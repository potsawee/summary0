import os
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle
# import numpy as np

from models.extractive import *
from data2 import ProcessedDocument

def train_extractive_model():
    print("Start training extractive model")

    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    data_type = 'test'
    MODEL_SAVE_DIR = "/home/alta/summary/pm574/summariser0/lib/trained_models/"

    use_gpu = True
    if use_gpu:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            # pdb.set_trace()
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '1' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    ext_sum = ExtractiveSummeriser(args, device)

    with open("lib/model_data/{}-{}.dat.pk.bin".format(data_type, args['max_pos_embed']), "rb") as f:
        data = pickle.load(f)

    # target_pos = load_extractive_labels(data_type, args['max_num_sentences'])
    with open("lib/model_data/target.{}.pk.bin".format(data_type), "rb") as f:
        target_pos = pickle.load(f)

    assert len(data) == len(target_pos), "len(data) != len(target_pos)"

    num_data = len(data)
    encoded_articles   = [None for i in range(num_data)]
    attention_masks    = [None for i in range(num_data)]
    token_type_ids_arr = [None for i in range(num_data)]
    cls_pos_arr        = [None for i in range(num_data)]

    for i, doc in enumerate(data):
        encoded_articles[i]   = doc.encoded_article
        attention_masks[i]    = doc.attention_mask
        token_type_ids_arr[i] = doc.token_type_ids
        cls_pos_arr[i]        = doc.cls_pos

    # Hyperparameters
    BATCH_SIZE = 10 # 3 for max_pos = 1024 & 8 for max_pos = 512
    NUM_EPOCHS = 2

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(ext_sum.parameters(), lr=2e-6 , betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    # zero the parameter gradients
    optimizer.zero_grad()

    # Initialisation: (1) TransformerLayer (2) Logistic Classification


    for epoch in range(NUM_EPOCHS):

        print("training epoch {}".format(epoch))
        num_batches = int(num_data / BATCH_SIZE) # deal with the last batch later
        print("num_batches = {}".format(num_batches))

        idx = 0

        # TODO:
        # 1. Try different optimisers e.g. PreSum optimizer
        # 2. Multiply the loss for the sentences that are ones (e.g. in the extractive summary)
        # 3. Not update the weights at every single step (batch_size = 3 only!)

        for bn in range(num_batches):
            # get my data
            inputs, targets, ms = get_a_batch(encoded_articles, attention_masks,
                                            token_type_ids_arr, cls_pos_arr,
                                            target_pos, args['max_num_sentences'],
                                            idx, BATCH_SIZE, device)

            mask = ms[0]
            lengths = ms[1]

            # forward + backward + optimize
            sent_scores = ext_sum(inputs)

            loss = criterion(sent_scores, targets)
            loss = (loss * mask.float()).sum() / lengths.sum()

            loss.backward()

            if bn % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            idx += BATCH_SIZE

            if bn % 10 == 0:
                print("batch number {}: loss = {}".format(bn, loss))
                print("sent_scores")
                print(sent_scores[0])
                print("target")
                print(targets[0])

            # if bn == 100:
            #     pdb.set_trace()

            # if bn % 1000 == 0:
            #     savepath = MODEL_SAVE_DIR + "extsum-test0-ep{}-bn{}.pt".format(epoch, bn)
            #     torch.save(model.state_dict(), savepath)

    print("Finish training extractive model")



def get_a_batch(encoded_articles, attention_masks,
                token_type_ids_arr, cls_pos_arr,
                target_pos, max_num_sentences,
                idx, batch_size, device):

    # input (x)
    input_ids = torch.tensor(encoded_articles[idx:idx+batch_size]).to(device)
    att_mask  = torch.tensor(attention_masks[idx:idx+batch_size]).to(device)
    tok_type_ids = torch.tensor(token_type_ids_arr[idx:idx+batch_size]).to(device)
    cls_pos = cls_pos_arr[idx:idx+batch_size]
    inputs = (input_ids, att_mask, tok_type_ids, cls_pos)

    # mask & lengths --- sentence-level
    sent_lengths = [len(cls) for cls in cls_pos]
    lengths = torch.tensor(sent_lengths, dtype=torch.long, device=device)
    mask = _sequence_mask(sequence_length=lengths, max_len=max_num_sentences)

    # target (y) --- shape = [batch_size, max_num_sentences]
    targets = torch.zeros((batch_size, max_num_sentences), dtype=torch.float32, device=device)
    for j in range(batch_size):
        pos = target_pos[idx+j]
        targets[j,pos] = targets[j,pos].fill_(1.0)
        # mask[j,pos] = mask[j,pos] * 10

    return inputs, targets, (mask, lengths)

def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()

    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = torch.autograd.Variable(seq_range_expand)

    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

train_extractive_model()
