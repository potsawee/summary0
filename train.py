import os
import sys
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
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"

    use_gpu = True
    if use_gpu:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            pdb.set_trace()
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '1' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    # Define the model
    ext_sum = ExtractiveSummeriser(args, device)

    # Load and prepare data
    train_data = load_data(args, 'train')
    val_data   = load_data(args, 'val')

    # Hyperparameters
    BATCH_SIZE = 10 # 3 for max_pos = 1024 | 10 for max_pos = 512
    NUM_EPOCHS = 5
    VAL_BATCH_SIZE = 200

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(ext_sum.parameters(), lr=2e-6 , betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    # zero the parameter gradients
    optimizer.zero_grad()

    # Initialisation: (1) TransformerLayer (2) Logistic Classification
    # done! => currently in __init__

    for epoch in range(NUM_EPOCHS):
        print("training epoch {}".format(epoch))
        num_batches = int(train_data['num_data'] / BATCH_SIZE) # deal with the last batch later
        print("num_batches = {}".format(num_batches))
        idx = 0
        for bn in range(num_batches):
            # get my data
            inputs, targets, ms = get_a_batch(train_data['encoded_articles'], train_data['attention_masks'],
                                            train_data['token_type_ids_arr'], train_data['cls_pos_arr'],
                                            train_data['target_pos'], args['max_num_sentences'],
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

            if bn % 2000 == 0:
                # -------------- Evaluate the model on validation data -------------- #
                print("evaluate the model at epoch {} step {}...".format(epoch, bn))
                num_val_epochs = int(val_data['num_data']/VAL_BATCH_SIZE)
                print("num_val_epochs = {}".format(num_val_epochs))

                ext_sum.eval() # switch to evaluation mode

                val_idx = 0
                val_total_loss = 0.0
                val_total_sentences = 0
                with torch.no_grad():
                    for _ in range(num_val_epochs):
                        val_inputs, val_targets, val_ms = get_a_batch(val_data['encoded_articles'],
                                                             val_data['attention_masks'], val_data['token_type_ids_arr'],
                                                             val_data['cls_pos_arr'], val_data['target_pos'],
                                                             args['max_num_sentences'], val_idx, VAL_BATCH_SIZE, device)
                        val_mask = val_ms[0]
                        val_lengths = val_ms[1]

                        val_sent_scores = ext_sum(val_inputs)
                        val_loss = criterion(val_sent_scores, val_targets)
                        val_total_loss += (val_loss * val_mask.float()).sum().data
                        val_total_sentences += val_lengths.sum().data

                        val_idx += VAL_BATCH_SIZE

                        print("#", end="")
                        sys.stdout.flush()

                avg_val_loss = val_total_loss / val_total_sentences
                print("\navg_val_loss_per_sentence = {}".format(avg_val_loss))

                ext_sum.train() # switch to training mode
                # ------------------------------------------------------------------ #

                # Save the model!
                savepath = args['model_save_dir'] + "extsum-AA1-ep{}-bn{}.pt".format(epoch, bn)
                torch.save(ext_sum.state_dict(), savepath)

        # do this when it finishes training an epoch

    print("Finish training extractive model")

def load_data(args, data_type):
    if data_type not in ['train', 'val', 'test']:
        raise ValueError('train/val/test only')

    path_data        = args['model_data_dir'] + "{}-{}.dat.pk.bin".format(data_type, args['max_pos_embed'])
    path_target      = args['model_data_dir'] + "target.{}-{}.pk.bin".format(data_type, args['max_num_sentences'])
    with open(path_data, "rb") as f: data = pickle.load(f)
    with open(path_target, "rb") as f: target_pos = pickle.load(f)

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

    data_dict = {
        'num_data': num_data,
        'encoded_articles': encoded_articles,
        'attention_masks': attention_masks,
        'token_type_ids_arr': token_type_ids_arr,
        'cls_pos_arr': cls_pos_arr,
        'target_pos': target_pos
    }

    return data_dict

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
    mask = generate_sequence_mask(sequence_length=lengths, max_len=max_num_sentences)

    # target (y) --- shape = [batch_size, max_num_sentences]
    targets = torch.zeros((batch_size, max_num_sentences), dtype=torch.float32, device=device)
    for j in range(batch_size):
        pos = target_pos[idx+j]
        targets[j,pos] = targets[j,pos].fill_(1.0)
        # mask[j,pos] = mask[j,pos] * 10

    return inputs, targets, (mask, lengths)

def generate_sequence_mask(sequence_length, max_len=None):
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

if __name__ == "__main__":
    train_extractive_model()
