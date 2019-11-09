import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle
from datetime import datetime

from models.extractive import *
from data2 import ProcessedDocument

def train_extractive_model():
    print("Start training extractive model")

    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "NOV9B"

    print("model_name = {}, max_num_sentences = {}, max_pos_embed = {}".format \
         (args['model_name'], args['max_num_sentences'], args['max_pos_embed']))

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
            os.environ["CUDA_VISIBLE_DEVICES"] = '3' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    # Define the model
    ext_sum = ExtractiveSummeriser(args, device)
    print(ext_sum)

    # Load and prepare data
    train_data = load_data(args, 'train')
    val_data   = load_data(args, 'val')

    # Hyperparameters
    BATCH_SIZE = 8 # 3 for max_pos = 1024 | 10 for max_pos = 512 | 8 for max_pos = 512 with validation
    NUM_EPOCHS = 10
    VAL_BATCH_SIZE = 200
    VAL_STOP_TRAINING = 5

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(ext_sum.parameters(), lr=2e-6 , betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    # zero the parameter gradients
    optimizer.zero_grad()

    # validation losses
    best_val_loss = 1e+10
    best_epoch = 0
    best_bn = 0
    stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        print("======================= Training epoch {} =======================".format(epoch))
        num_batches = int(train_data['num_data'] / BATCH_SIZE) + 1 # plus 1 for the last batch
        print("num_batches = {}".format(num_batches))
        idx = 0
        for bn in range(num_batches):

            # check if it is the last batch
            if bn == (num_batches - 1): last_batch = True
            else: last_batch = False

            # get my data
            inputs, targets, ms = get_a_batch(train_data['encoded_articles'], train_data['attention_masks'],
                                            train_data['token_type_ids_arr'], train_data['cls_pos_arr'],
                                            train_data['target_pos'], args['max_num_sentences'],
                                            idx, BATCH_SIZE, last_batch, device)
            mask = ms[0]
            lengths = ms[1]

            # forward + backward + optimize
            sent_scores = ext_sum(inputs)

            loss = criterion(sent_scores, targets)
            loss = (loss * mask.float()).sum() / lengths.sum()
            loss.backward()

            idx += BATCH_SIZE

            if bn % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if bn % 20 == 0:
                print("[{}] batch number {}/{}: loss = {}".format(str(datetime.now()), bn, num_batches, loss))
                sys.stdout.flush()

            if bn % 2000 == 0:
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at epoch {} step {}".format(epoch, bn))
                ext_sum.eval() # switch to evaluation mode
                with torch.no_grad():
                    avg_val_loss = evaluate(ext_sum, val_data, VAL_BATCH_SIZE, args, device)
                print("avg_val_loss_per_sentence = {}".format(avg_val_loss))
                ext_sum.train() # switch to training mode

                # ------------------- Save the model OR Stop training ------------------- #
                if avg_val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_bn = bn
                    savepath = args['model_save_dir']+"extsum-{}-ep{}-bn{}.pt".format(args['model_name'],epoch,bn)
                    torch.save(ext_sum.state_dict(), savepath)
                    print("Model improved & saved at {}".format(savepath))
                else:
                    print("Model not improved #{}".format(stop_counter))
                    if stop_counter < VAL_STOP_TRAINING:
                        # load the previous model
                        latest_model = args['model_save_dir']+"extsum-{}-ep{}-bn{}.pt".format(args['model_name'],best_epoch,best_bn)
                        ext_sum.load_state_dict(torch.load(latest_model))
                        ext_sum.train()
                        print("Restored model from {}".format(latest_model))
                        stop_counter += 1
                    else:
                        print("Model has not improved for {} times! Stop training.".format(VAL_STOP_TRAINING))
                        return

    print("End of training extractive model")

def evaluate(model, eval_data, eval_batch_size, args, device):
    # print("evaluate the model at epoch {} step {}...".format(epoch, bn))
    num_eval_epochs = int(eval_data['num_data']/eval_batch_size) + 1
    print("num_eval_epochs = {}".format(num_eval_epochs))

    eval_idx = 0
    eval_total_loss = 0.0
    eval_total_sentences = 0

    criterion = nn.BCELoss(reduction='none')

    # with torch.no_grad():
    for bn in range(num_eval_epochs):

        # check if it is the last batch
        if bn == (num_eval_epochs - 1): last_batch = True
        else: last_batch = False

        eval_inputs, eval_targets, eval_ms = get_a_batch(eval_data['encoded_articles'],
                                             eval_data['attention_masks'], eval_data['token_type_ids_arr'],
                                             eval_data['cls_pos_arr'], eval_data['target_pos'],
                                             args['max_num_sentences'], eval_idx, eval_batch_size,
                                             last_batch, device)
        eval_mask = eval_ms[0]
        eval_lengths = eval_ms[1]

        eval_sent_scores = model(eval_inputs)
        eval_loss = criterion(eval_sent_scores, eval_targets)
        eval_total_loss += (eval_loss * eval_mask.float()).sum().item()
        eval_total_sentences += eval_lengths.sum().item()

        eval_idx += eval_batch_size

        print("#", end="")
        sys.stdout.flush()

    print('\n')
    avg_eval_loss = eval_total_loss / eval_total_sentences

    return avg_eval_loss

def load_data(args, data_type):
    if data_type not in ['train', 'val', 'test']:
        raise ValueError('train/val/test only')

    path_data        = args['model_data_dir'] + "{}-{}.dat.nltk.pk.bin".format(data_type, args['max_pos_embed'])
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
                idx, batch_size, last_batch, device):

    if last_batch == True:
        # print("the last batch is fetched")
        num_data = len(encoded_articles)
        batch_size = num_data - idx

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
