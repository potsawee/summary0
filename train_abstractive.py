import os
import sys
import torch
import pdb
import pickle

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
    args['update_nbatches'] = 4
    args['batch_size'] = 8 # 3 for max_pos = 1024 | 10 for max_pos = 512 | 8 for max_pos = 512 with validation
    args['num_epochs'] = 10
    args['val_batch_size'] = 200
    args['val_stop_training'] = 10
    args['random_seed'] = 28
    args['lr'] = 5e-6
    args['adjust_lr'] = True
    # ---------------------------------------------------------------------------------- #
    args['use_gpu'] = False
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "NOV14dev"
    # load_model: None or specify path e.g. "/home/alta/summary/pm574/summariser0/lib/trained_models/best_NOV9.pt"
    # args['load_model'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-NOV13F-ep1-bn0.pt"
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
            os.environ["CUDA_VISIBLE_DEVICES"] = '0' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    data    = load_data(args, 'test')
    summary = load_summary(args, 'test')
    assert data['num_data'] == summary['num_data'], "data['num_data'] != summary['num_data']"

    abs_sum = AbstractiveSummariser(args, device=device)

    criterion = nn.NLLLoss(reduction='none')

    num_batches = int(data['num_data']/args['batch_size']) + 1
    idx = 0
    for bn in range(num_batches):
        batch = get_a_batch_abs(data['encoded_articles'], data['attention_masks'],
                            data['token_type_ids_arr'], data['cls_pos_arr'],
                            data['target_pos'], args['max_num_sentences'],
                            summary['encoded_abstracts'], summary['abs_lengths'],
                            args['max_summary_length'],
                            idx, args['batch_size'], False, device)

        output = abs_sum(batch)
        pdb.set_trace()


    print("End of training extractive model")

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
    mem_mask = ~mem_mask.bool()
    memory_key_padding_mask = torch.tensor(mem_mask.data, dtype=KEYPADMASK_DTYPE).to(device)

    # input to the decoder
    tgt_ids = torch.tensor(encoded_abstracts[idx:idx+batch_size]).to(device)
    key_padding_mask = [None for _ in range(batch_size)]
    for j in range(batch_size):
        key_padding_mask[j] = [False]*abs_lengths[idx+j]+[True]*(max_summary_length-abs_lengths[idx+j])
    tgt_key_padding_mask = torch.tensor(key_padding_mask, dtype=KEYPADMASK_DTYPE).to(device)

    batch = {
        'encoder': (enc_inputs, enc_targets, enc_key_padding_mask, ms),
        'memory':  memory_key_padding_mask,
        'decoder': (tgt_ids, tgt_key_padding_mask)
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
