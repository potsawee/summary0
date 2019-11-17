import os
import sys
import torch
import torch.optim as optim
import pdb
import pickle
import random
from datetime import datetime

from models.abstractive import *
from data2 import ProcessedDocument, ProcessedSummary
from train import load_data, get_a_batch

if   torch.__version__ == '1.1.0': KEYPADMASK_DTYPE = torch.uint8
elif torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else: raise Exception("Torch Version not supoorted")

def train_abstractive_model():
    print("Start training abstractive model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['eval_nbatches'] = 2000
    args['update_nbatches'] = 5
    args['batch_size'] = 6
    args['num_epochs'] = 10
    args['val_batch_size'] = 100
    args['val_stop_training'] = 10
    args['random_seed'] = 28
    args['lr'] = 5e-6
    args['adjust_lr'] = True
    # ---------------------------------------------------------------------------------- #
    args['use_gpu'] = True
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "ANOV17A"
    # load_model: None or specify path e.g. "/home/alta/summary/pm574/summariser0/lib/trained_models/best_NOV9.pt"
    # args['load_model'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/abssum-NOV13F-ep1-bn0.pt"
    args['load_model'] = None
    args['best_val_loss'] = 1e+10
    # ---------------------------------------------------------------------------------- #
    args['max_summary_length'] = 96
    # ---------------------------------------------------------------------------------- #

    if args['use_gpu']:
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

    train_data    = load_data(args, 'trainx')
    train_summary = load_summary(args, 'trainx')
    val_data      = load_data(args, 'val')
    val_summary   = load_summary(args, 'val')

    # random seed
    random.seed(args['random_seed'])

    assert train_data['num_data'] == train_summary['num_data'], \
        "train_data['num_data'] != train_summary['num_data']"

    abs_sum = AbstractiveSummariser(args, device=device)
    print(abs_sum)

    # Hyperparameters
    BATCH_SIZE = args['batch_size']
    NUM_EPOCHS = args['num_epochs']
    VAL_BATCH_SIZE = args['val_batch_size']
    VAL_STOP_TRAINING = args['val_stop_training']

    vocab_size = abs_sum.decoder.linear_decoder.out_features

    criterion = nn.NLLLoss(reduction='none')
    optimizer = optim.Adam(abs_sum.parameters(), lr=args['lr'] , betas=(0.9,0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    # validation losses
    best_val_loss = args['best_val_loss']
    best_epoch = 0
    best_bn = 0
    stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        print("======================= Training epoch {} =======================".format(epoch))
        num_batches = int(train_data['num_data']/BATCH_SIZE) + 1
        print("num_batches = {}".format(num_batches))

        # Random shuffle the training data
        train_data, train_summary = shuffle_data2(train_data, train_summary)

        idx = 0

        for bn in range(num_batches):
            # check if it is the last batch
            if bn == (num_batches - 1): last_batch = True
            else: last_batch = False

            batch = get_a_batch_abs(train_data['encoded_articles'], train_data['attention_masks'],
                                train_data['token_type_ids_arr'], train_data['cls_pos_arr'],
                                train_data['target_pos'], args['max_num_sentences'],
                                train_summary['encoded_abstracts'], train_summary['abs_lengths'],
                                args['max_summary_length'],
                                idx, BATCH_SIZE, last_batch, device)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(batch['decoder'])
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            # forward + backward
            decoder_output = abs_sum(batch)

            loss = criterion(decoder_output.view(-1, vocab_size), decoder_target)
            loss = (loss * decoder_mask).sum() /  decoder_mask.sum()
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
                abs_sum.eval() # switch to evaluation mode
                with torch.no_grad():
                    avg_val_loss = evaluate2(abs_sum, val_data, val_summary, VAL_BATCH_SIZE, vocab_size, args, device)
                print("avg_val_loss_per_token = {}".format(avg_val_loss))
                abs_sum.train() # switch to training mode
                # ------------------- Save the model OR Stop training ------------------- #
                if avg_val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_bn = bn
                    savepath = args['model_save_dir']+"abssum-{}-ep{}-bn{}.pt".format(args['model_name'],epoch,bn)
                    torch.save(abs_sum.state_dict(), savepath)
                    print("Model improved & saved at {}".format(savepath))
                else:
                    print("Model not improved #{}".format(stop_counter))
                    if stop_counter < VAL_STOP_TRAINING:
                        # load the previous model
                        latest_model = args['model_save_dir']+"abssum-{}-ep{}-bn{}.pt".format(args['model_name'],best_epoch,best_bn)
                        abs_sum.load_state_dict(torch.load(latest_model))
                        abs_sum.train()
                        print("Restored model from {}".format(latest_model))
                        stop_counter += 1
                    else:
                        print("Model has not improved for {} times! Stop training.".format(VAL_STOP_TRAINING))
                        return

    print("End of training abstractive model")

def shift_decoder_target(batch_decoder):
    # MASK_TOKEN_ID = 103
    batch_size = batch_decoder[0].size(0)
    max_len = batch_decoder[0].size(1)
    dtype0  = batch_decoder[0].dtype
    dtype1  = batch_decoder[-1].dtype
    device  = batch_decoder[0].device.type

    decoder_target = torch.zeros((batch_size, max_len), dtype=dtype0, device=device)
    decoder_mask   = torch.zeros((batch_size, max_len), dtype=dtype1, device=device)

    decoder_target[:,:-1] = batch_decoder[0].clone().detach()[:,1:]
    decoder_mask[:,:-1]  = batch_decoder[-1].clone().detach()[:,1:]

    decoder_target[:,-1:] = 103 # MASK_TOKEN_ID = 103
    # decoder_mask[:,-1:] = 0.0 # ---> already filled with 0.0

    return decoder_target, decoder_mask

def evaluate2(model, eval_data, eval_summary, eval_batch_size, vocab_size, args, device):
    num_eval_epochs = int(eval_data['num_data']/eval_batch_size) + 1
    print("num_eval_epochs = {}".format(num_eval_epochs))

    eval_idx = 0
    eval_total_loss = 0.0
    eval_total_tokens = 0

    criterion = nn.NLLLoss(reduction='none')

    for bn in range(num_eval_epochs):
        # check if it is the last batch
        if bn == (num_eval_epochs - 1): last_batch = True
        else: last_batch = False

        batch = get_a_batch_abs(
                eval_data['encoded_articles'], eval_data['attention_masks'],
                eval_data['token_type_ids_arr'], eval_data['cls_pos_arr'],
                eval_data['target_pos'], args['max_num_sentences'],
                eval_summary['encoded_abstracts'], eval_summary['abs_lengths'],
                args['max_summary_length'], eval_idx, eval_batch_size,
                last_batch, device)

        # decoder target
        decoder_target = batch['decoder'][0].view(-1)
        decoder_mask   = batch['decoder'][-1].view(-1)

        # forward + backward
        decoder_output = model(batch)
        loss = criterion(decoder_output.view(-1, vocab_size), decoder_target)
        eval_total_loss += (loss * decoder_mask).sum().item()
        eval_total_tokens += decoder_mask.sum().item()

        eval_idx += eval_batch_size

        print("#", end="")
        sys.stdout.flush()

    print('\n')
    avg_eval_loss = eval_total_loss / eval_total_tokens

    return avg_eval_loss

def shuffle_data2(data_dict, summary_dict):
    assert data_dict['num_data'] == summary_dict['num_data'], \
        "data_dict['num_data'] != summary_dict['num_data']"

    _x = list(zip(
        data_dict['encoded_articles'],
        data_dict['attention_masks'],
        data_dict['token_type_ids_arr'],
        data_dict['cls_pos_arr'],
        data_dict['target_pos'],
        summary_dict['encoded_abstracts'],
        summary_dict['abs_lengths']
    ))

    random.shuffle(_x)
    x1, x2, x3, x4, x5, y1, y2 = zip(*_x)

    shuffled_data_dict = {
        'num_data': data_dict['num_data'],
        'encoded_articles':   x1,
        'attention_masks':    x2,
        'token_type_ids_arr': x3,
        'cls_pos_arr':        x4,
        'target_pos':         x5
    }
    shuffled_summary_dict = {
        'num_data': summary_dict['num_data'],
        'encoded_abstracts':  y1,
        'abs_lengths':        y2,
    }
    return shuffled_data_dict, shuffled_summary_dict

def get_a_batch_abs(encoded_articles, attention_masks,
                token_type_ids_arr, cls_pos_arr,
                target_pos, max_num_sentences,
                encoded_abstracts, abs_lengths,
                max_summary_length,
                idx, batch_size, last_batch, device):

    if last_batch == True:
        num_data = len(encoded_articles)
        batch_size = num_data - idx

    # input to the encoder
    enc_inputs, enc_targets, enc_key_padding_mask, ms = \
        get_a_batch(encoded_articles, attention_masks,
                    token_type_ids_arr, cls_pos_arr,
                    target_pos, max_num_sentences,
                    idx, batch_size, last_batch, device)

    # enc_key_padding_mask => the mask for input into the transformer for extractive task
    # memory_key_padding_mask => the mask for decoder (enc-dec attention) for abstractive task
    # memory_key_padding_mask => shape = [batch_size, max_pos_embed]
    mem_mask = enc_inputs[1].clone() # bert_attn_mask

    if KEYPADMASK_DTYPE  == torch.uint8: mem_mask ^= 1
    elif KEYPADMASK_DTYPE == torch.bool: mem_mask = ~mem_mask.bool()
    else: raise Exception("Torch Version not supoorted")

    memory_key_padding_mask = torch.tensor(mem_mask.data, dtype=KEYPADMASK_DTYPE).to(device)

    # input to the decoder
    tgt_ids = torch.tensor(encoded_abstracts[idx:idx+batch_size]).to(device)
    key_padding_mask = [None for _ in range(batch_size)]
    decoder_mask     = [None for _ in range(batch_size)]
    for j in range(batch_size):
        key_padding_mask[j] = [False]*abs_lengths[idx+j]+[True]*(max_summary_length-abs_lengths[idx+j])
        decoder_mask[j]     = [1.0]*abs_lengths[idx+j]+[0.0]*(max_summary_length-abs_lengths[idx+j])
    tgt_key_padding_mask = torch.tensor(key_padding_mask, dtype=KEYPADMASK_DTYPE).to(device)
    decoder_mask = torch.tensor(decoder_mask, dtype=torch.float).to(device)


    batch = {
        'encoder': (enc_inputs, enc_targets, enc_key_padding_mask, ms),
        'memory':  memory_key_padding_mask,
        'decoder': (tgt_ids, tgt_key_padding_mask, decoder_mask)
    }

    return batch

def load_summary(args, data_type):
    if data_type not in ['train', 'val', 'test', 'trainx']:
        raise ValueError('train/val/test only')

    path = args['model_data_dir'] + "abstract.{}-{}.pk.bin".format(data_type, args['max_summary_length'])
    with open(path, "rb") as f: summaries = pickle.load(f)

    N = len(summaries)
    encoded_abstracts = [None for i in range(N)]
    abs_lengths       = [None for i in range(N)]

    for i, sum in enumerate(summaries):
        encoded_abstracts[i] = sum.encoded_abstract
        abs_lengths[i]       = sum.length

    summary_dict = {
        'num_data': N,
        'encoded_abstracts': encoded_abstracts,
        'abs_lengths': abs_lengths
    }
    return summary_dict


if __name__ == "__main__":
    train_abstractive_model()
