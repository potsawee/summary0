import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle
import random
from datetime import datetime

from models.extractive import *
from data2 import ProcessedDocument

if   torch.__version__ == '1.1.0': KEYPADMASK_DTYPE = torch.uint8
elif torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else: raise Exception("Torch Version not supoorted")

def train_extractive_model():
    print("Start training extractive model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['eval_nbatches'] = 2000
    args['update_nbatches'] = 4
    args['batch_size'] = 8 # 3 for max_pos = 1024 | 10 for max_pos = 512 | 8 for max_pos = 512 with validation
    args['num_epochs'] = 10
    args['val_batch_size'] = 200
    args['val_stop_training'] = 10
    args['random_seed'] = 30
    args['lr'] = 5e-6
    args['adjust_lr'] = True
    # ---------------------------------------------------------------------------------- #
    args['use_gpu'] = True
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "NOV15A"
    # load_model: None or specify path e.g. "/home/alta/summary/pm574/summariser0/lib/trained_models/best_NOV9.pt"
    args['load_model'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-NOV13Fc-ep0-bn4000.pt"
    # args['load_model'] = None
    args['best_val_loss'] = 1e+10
    # ---------------------------------------------------------------------------------- #

    print("model_name = {}, max_num_sentences = {}, max_pos_embed = {}".format \
         (args['model_name'], args['max_num_sentences'], args['max_pos_embed']))

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            # pdb.set_trace()
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '2' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    # Define the model
    ext_sum = ExtractiveSummariser(args, device)
    print(ext_sum)

    # Load model if specified (path to pytorch .pt)
    if args['load_model'] != None:
        ext_sum.load_state_dict(torch.load(args['load_model']))
        ext_sum.train()
        print("Loaded model from {}".format(args['load_model']))
    else:
        print("Train a new model")

    # Load and prepare data
    train_data = load_data(args, 'trainx')
    val_data   = load_data(args, 'val')

    # random seed
    random.seed(args['random_seed'])

    # Hyperparameters
    BATCH_SIZE = args['batch_size']
    NUM_EPOCHS = args['num_epochs']
    VAL_BATCH_SIZE = args['val_batch_size']
    VAL_STOP_TRAINING = args['val_stop_training']

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(ext_sum.parameters(), lr=args['lr'] , betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    # zero the parameter gradients
    optimizer.zero_grad()

    # validation losses
    best_val_loss = args['best_val_loss']
    best_epoch = 0
    best_bn = 0
    stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        print("======================= Training epoch {} =======================".format(epoch))
        num_batches = int(train_data['num_data'] / BATCH_SIZE) + 1 # plus 1 for the last batch
        print("num_batches = {}".format(num_batches))

        # Random shuffle the training data
        train_data = shuffle_data(train_data)

        idx = 0

        for bn in range(num_batches):
            # adjust the learning rate of the optimizer
            if args['adjust_lr']:
                adjust_lr(optimizer, epoch, epoch_size=num_batches, bn=bn, warmup=10000)

            # check if it is the last batch
            if bn == (num_batches - 1): last_batch = True
            else: last_batch = False

            # get my data
            inputs, targets, key_padding_mask, ms  = \
                get_a_batch(train_data['encoded_articles'], train_data['attention_masks'],
                            train_data['token_type_ids_arr'], train_data['cls_pos_arr'],
                            train_data['target_pos'], args['max_num_sentences'],
                            idx, BATCH_SIZE, last_batch, device)
            mask = ms[0]
            lengths = ms[1]

            # forward + backward + optimize
            sent_scores = ext_sum(inputs, key_padding_mask)

            loss = criterion(sent_scores, targets)
            loss = (loss * mask.float()).sum() / lengths.sum()
            loss.backward()

            idx += BATCH_SIZE

            if bn % args['update_nbatches'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            if bn % 20 == 0:
                print("[{}] batch number {}/{}: loss = {}".format(str(datetime.now()), bn, num_batches, loss))
                sys.stdout.flush()

            if bn % args['eval_nbatches'] == 0: # e.g. eval every 2000 batches
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at epoch {} step {}".format(epoch, bn))
                print("learning_rate = {}".format(optimizer.param_groups[0]['lr']))
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

def adjust_lr(optimizer, epoch, epoch_size, bn, warmup=10000):
    """to adjust the learning rate"""
    step = (epoch * epoch_size) + bn + 1 # plus 1 to avoid ZeroDivisionError
    lr = 2e-3 * min(step**(-0.5), step*(warmup**(-1.5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return

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

        eval_inputs, eval_targets, keenc_inputsy_padding_mask, eval_ms = \
            get_a_batch(eval_data['encoded_articles'],
                        eval_data['attention_masks'], eval_data['token_type_ids_arr'],
                        eval_data['cls_pos_arr'], eval_data['target_pos'],
                        args['max_num_sentences'], eval_idx, eval_batch_size,
                        last_batch, device)

        eval_mask = eval_ms[0]
        eval_lengths = eval_ms[1]

        eval_sent_scores = model(eval_inputs, key_padding_mask)
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
    if data_type not in ['train', 'val', 'test', 'trainx']:
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

def shuffle_data(data_dict):
    # data_dict generated by load_data
    _x = list(zip(
        data_dict['encoded_articles'],
        data_dict['attention_masks'],
        data_dict['token_type_ids_arr'],
        data_dict['cls_pos_arr'],
        data_dict['target_pos']
    ))

    random.shuffle(_x)
    x1, x2, x3, x4, x5 = zip(*_x)

    shuffled_data_dict = {
        'num_data': data_dict['num_data'],
        'encoded_articles':   x1,
        'attention_masks':    x2,
        'token_type_ids_arr': x3,
        'cls_pos_arr':        x4,
        'target_pos':         x5
    }
    return shuffled_data_dict

def get_a_batch(encoded_articles, attention_masks,
                token_type_ids_arr, cls_pos_arr,
                target_pos, max_num_sentences,
                idx, batch_size, last_batch, device,
                abstractive_task=False):

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

    # (encoder_output) key_padding_mask
    # enc_key_padding_mask => (batch_size, num_sentences)
    enc_key_padding_mask = [None for _ in range(batch_size)]
    for i in range(batch_size):
        slen = len(cls_pos[i])
        if slen <= max_num_sentences:
            enc_key_padding_mask[i] = [False]*slen + [True]*(max_num_sentences-slen)
        else:
            enc_key_padding_mask[i] = [False]*max_num_sentences
    enc_key_padding_mask = torch.tensor(enc_key_padding_mask, dtype=KEYPADMASK_DTYPE).to(device)


    # mask & lengths --- sentence-level (for computing loss)
    sent_lengths = [len(cls) for cls in cls_pos]
    lengths = torch.tensor(sent_lengths, dtype=torch.long, device=device)
    mask = generate_sequence_mask(sequence_length=lengths, max_len=max_num_sentences)

    # target (y) --- shape = [batch_size, max_num_sentences]
    targets = torch.zeros((batch_size, max_num_sentences), dtype=torch.float32, device=device)
    for j in range(batch_size):
        pos = target_pos[idx+j]
        targets[j,pos] = targets[j,pos].fill_(1.0)
        # mask[j,pos] = mask[j,pos] * 10

    return inputs, targets, enc_key_padding_mask, (mask, lengths)

def generate_sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()

    batch_size = sequence_length.size(0)
    # seq_range = torch.range(0, max_len - 1).long() ---> torch.range is depricated
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = torch.autograd.Variable(seq_range_expand)

    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

if __name__ == "__main__":
    train_extractive_model()
