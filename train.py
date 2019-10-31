import os
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle

from models.extractive import *
from data2 import ProcessedDocument

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

    args = None

    ext_sum = ExtractiveSummeriser(args, device)

    # input_ids, attention_mask, token_type_ids, position_ids, cls_pos = create_dummy_input()
    # inputs = (input_ids, attention_mask, token_type_ids, position_ids, cls_pos)
    # sent_scores = ext_sum(inputs, cls_pos)
    # target = torch.zeros((50, 100), dtype=torch.float32)

    with open("test31oct.dat.pk.bin", "rb") as f:
        data = pickle.load(f)

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

    input_ids = torch.tensor(encoded_articles[:4]).to(device)
    att_mask  = torch.tensor(attention_masks[:4]).to(device)
    tok_type_ids = torch.tensor(token_type_ids_arr[:4]).to(device)
    cls_pos = cls_pos_arr[:4]

    inputs = (input_ids, att_mask, tok_type_ids, cls_pos)

    sent_scores = ext_sum(inputs)

    pdb.set_trace()

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ext_sum.parameters(), lr=0.001 , betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    for epoch in range(2):
        running_loss = 0.0
        for i, batch in enumerate(batches):
            # get my data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            sent_scores = ext_sum(inputs, cls_pos)
            loss = criterion(sent_scores, target)
            print("before backward")
            loss.backward()
            print("after backward")

            optimizer.step()
            print("hi")



    print("Finish training extractive model")

train_extractive_model()
