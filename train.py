import os
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle

from models.extractive import *
from data2 import ProcessedDocument, load_extractive_labels

def train_extractive_model():
    print("Start training extractive model")

    use_gpu = True
    if use_gpu:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '3' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    args = {}
    args['max_num_sentences'] = 32

    ext_sum = ExtractiveSummeriser(args, device)

    # input_ids, attention_mask, token_type_ids, position_ids, cls_pos = create_dummy_input()
    # inputs = (input_ids, attention_mask, token_type_ids, position_ids, cls_pos)
    # sent_scores = ext_sum(inputs, cls_pos)
    # target = torch.zeros((50, 100), dtype=torch.float32)

    with open("test31oct.dat.pk.bin", "rb") as f:
        data = pickle.load(f)

    target_pos = load_extractive_labels("test", args['max_num_sentences'])

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
    BATCH_SIZE = 3
    NUM_EPOCHS = 2

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ext_sum.parameters(), lr=0.001 , betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    for epoch in range(NUM_EPOCHS):
        print("training epoch {}".format(epoch))
        running_loss = 0.0
        num_batches = int(num_data / BATCH_SIZE) # deal with the last batch later
        idx = 0
        for bn in range(num_batches):
            # get my data
            inputs, targets = get_a_batch(encoded_articles, attention_masks,
                                          token_type_ids_arr, cls_pos_arr,
                                          target_pos, args['max_num_sentences'],
                                          idx, BATCH_SIZE, device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            sent_scores = ext_sum(inputs)

            loss = criterion(sent_scores, targets)

            loss.backward()
            optimizer.step()

            idx += BATCH_SIZE

            if bn % 10 == 0:
                print("batch number {}: loss = {}".format(bn, loss))

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

    # target (y) --- shape = [batch_size, max_num_sentences]
    targets = torch.zeros((batch_size, max_num_sentences), dtype=torch.float32, device=device)
    for j in range(batch_size):
        pos = target_pos[idx+j]
        targets[j,pos] = targets[j,pos].fill_(1.0)

    # a batch = (inputs, targets)
    return inputs, targets

train_extractive_model()
