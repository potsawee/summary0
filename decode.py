"""Inference time script for the abstractive task"""
import os
import sys
import torch
import pdb
import numpy as np
from datetime import datetime

from transformers import BertTokenizer

from models.abstractive import *
from train import load_data
from train_abstractive import get_a_batch_abs
from data2 import ProcessedDocument, ProcessedSummary

if torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else: raise Exception("Torch Version not supoorted")

START_TOKEN = '[CLS]'
SEP_TOKEN   = '[SEP]'
STOP_TOKEN  = '[MASK]'

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

START_TOKEN_ID = bert_tokenizer.convert_tokens_to_ids(START_TOKEN)
SEP_TOKEN_ID   = bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
STOP_TOKEN_ID  = bert_tokenizer.convert_tokens_to_ids(STOP_TOKEN)

TEST_DATA_SIZE = 11490
VOCAB_SIZE     = 30522

def beam_search(model, data, args, start_idx, batch_size, num_batches, k):
    device = args['device']
    max_summary_length = args['max_summary_length']
    time_step = max_summary_length
    idx = 0
    summary_out_dir = args['summary_out_dir']


    for bn in range(num_batches):
        if (start_idx+idx+batch_size) > (TEST_DATA_SIZE-1): # last file = 11489
            last_batch = True
            summaries = [None for _ in range(TEST_DATA_SIZE - (start_idx+idx))]
            batch_size = TEST_DATA_SIZE - (start_idx+idx)
        else:
            summaries = [None for _ in range(batch_size)]
            last_batch = False

        beams = [None for _ in range(k)]
        beam_scores = np.zeros((batch_size, k))

        batch = get_a_batch_abs(
            data['encoded_articles'], data['attention_masks'],
            data['token_type_ids_arr'], data['cls_pos_arr'],
            data['target_pos'], args['max_num_sentences'],
            None, None, None, idx+start_idx, batch_size,
            last_batch, device, decoding=True)
        # need to construct the input to the decoder
        tgt_ids = torch.zeros((batch_size, max_summary_length), dtype=torch.int64).to(device)
        for t in range(time_step-1):
            # the first token is '[CLS]'
            if t == 0:
                tgt_ids[:,0] = START_TOKEN_ID
                for i in range(k):
                    beams[i] = tgt_ids

            # tgt_key_padding_mask
            row_padding_mask = [False]*(t+1) + [True]*(max_summary_length-t-1)
            padding_mask     = [row_padding_mask for _ in range(batch_size)]
            tgt_key_padding_mask = torch.tensor(padding_mask, dtype=KEYPADMASK_DTYPE).to(device)

            decoder_output_t_array = torch.zeros((batch_size, k*VOCAB_SIZE)) # output at a time step * beam_width

            for i, beam in enumerate(beams):
                batch['decoder'] = (beam, tgt_key_padding_mask, None)
                # decoder_output => [batch_size, max_summary_length, vocab_size]
                decoder_output = model(batch)
                decoder_output_t_array[:,i*VOCAB_SIZE:(i+1)*VOCAB_SIZE] = decoder_output[:,t,:]
                # add previous beam score bias
                for n_idx in range(batch_size):
                    decoder_output_t_array[n_idx,i*VOCAB_SIZE:(i+1)*VOCAB_SIZE] += beam_scores[n_idx,i]
                if t == 0: break # only fill once for the first time step


            # scores, indices => [batch_size, k]
            scores, indices = torch.topk(decoder_output_t_array, k=k, dim=-1)
            new_beams = [torch.zeros((batch_size, max_summary_length), dtype=torch.int64).to(device) for _ in range(k)]
            for r_idx, row in enumerate(indices):
                for c_idx, node in enumerate(row):
                    vocab_idx = node % VOCAB_SIZE
                    beam_idx  = int(node / VOCAB_SIZE)

                    new_beams[c_idx][r_idx,:t+1] = beams[beam_idx][r_idx,:t+1]
                    new_beams[c_idx][r_idx,t+1]  = vocab_idx


            beam_scores = scores.cpu().numpy()
            beams = new_beams

        # finish t = 0,...,max_summary_length
        for j in range(batch_size):
            # summaries[j] = tgtids2summary(tgt_ids[j].cpu().numpy())
            summaries[j] = tgtids2summary(beams[0][j].cpu().numpy())

        write_summary_files(summary_out_dir, summaries, start_idx+idx)

        print("[{}] batch {}/{} --- idx [{},{})".format(
                str(datetime.now()), bn+1, num_batches,
                start_idx+idx, start_idx+idx+batch_size))
        sys.stdout.flush()
        idx += batch_size

def greedy_search(model, data, args, start_idx, batch_size, num_batches):
    # decode idx from [start_idx, end_idx)
    # model = trained PyTorch abstractive model
    # data  = retrieved from load_data
    device = args['device']
    max_summary_length = args['max_summary_length']
    time_step = max_summary_length
    idx = 0
    summary_out_dir = args['summary_out_dir']

    for bn in range(num_batches):
        if (start_idx+idx+batch_size) > (TEST_DATA_SIZE-1): # last file = 11489
            last_batch = True
            summaries = [None for _ in range(TEST_DATA_SIZE - (start_idx+idx))]
            batch_size = TEST_DATA_SIZE - (start_idx+idx)
        else:
            summaries = [None for _ in range(batch_size)]
            last_batch = False
        batch = get_a_batch_abs(
            data['encoded_articles'], data['attention_masks'],
            data['token_type_ids_arr'], data['cls_pos_arr'],
            data['target_pos'], args['max_num_sentences'],
            None, None, None, idx+start_idx, batch_size,
            last_batch, device, decoding=True)
        # need to construct the input to the decoder
        tgt_ids = torch.zeros((batch_size, max_summary_length), dtype=torch.int64).to(device)
        for t in range(time_step-1):
            # the first token is '[CLS]'
            if t == 0: tgt_ids[:,0] = START_TOKEN_ID
            # tgt_key_padding_mask
            row_padding_mask = [False]*(t+1) + [True]*(max_summary_length-t-1)
            padding_mask     = [row_padding_mask for _ in range(batch_size)]
            tgt_key_padding_mask = torch.tensor(padding_mask, dtype=KEYPADMASK_DTYPE).to(device)

            batch['decoder'] = (tgt_ids, tgt_key_padding_mask, None)
            # decoder_output => [batch_size, max_summary_length, vocab_size]
            decoder_output = model(batch)
            output_at_t = torch.argmax(decoder_output, dim=-1)[:,t]
            tgt_ids[:,t+1] = output_at_t

        for j in range(batch_size):
            summaries[j] = tgtids2summary(tgt_ids[j].cpu().numpy())
        write_summary_files(summary_out_dir, summaries, start_idx+idx)
        print("[{}] batch {}/{} --- idx [{},{})".format(
                str(datetime.now()), bn+1, num_batches,
                start_idx+idx, start_idx+idx+batch_size))
        sys.stdout.flush()
        idx += batch_size
    return

def write_summary_files(dir, summaries, start_idx):
    if not os.path.exists(dir): os.makedirs(dir)
    num_data = len(summaries)
    for idx in range(num_data):
        filepath = dir + 'file.{}.txt'.format(idx+start_idx)
        line = '\n'.join(summaries[idx])
        with open(filepath, 'w') as f:
            f.write(line)


def tgtids2summary(tgt_ids):
    # tgt_ids = a row of numpy array containing token ids
    bert_decoded = bert_tokenizer.decode(tgt_ids)
    # truncate START_TOKEN & part after STOP_TOKEN
    stop_idx = bert_decoded.find(STOP_TOKEN)
    processed_bert_decoded = bert_decoded[5:stop_idx]
    summary = [s.strip() for s in processed_bert_decoded.split(SEP_TOKEN)]
    return summary

def decode(start_idx):
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['max_summary_length'] = 96
    args['use_gpu'] = True
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "ANOV19B"
    args['model_epoch'] = 0
    args['model_bn'] = 0
    args['decoding_method'] = 'beamsearch'
    # ---------------------------------------------------------------------------------- #
    args['summary_out_dir'] = \
    '/home/alta/summary/pm574/summariser0/out_summary/abstractive/model-{}-ep{}-bn{}-{}/' \
    .format(args['model_name'], args['model_epoch'], args['model_bn'], args['decoding_method'])
    # ---------------------------------------------------------------------------------- #
    start_idx = start_idx
    batch_size = 4
    num_batches = 5
    # ---------------------------------------------------------------------------------- #

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
            os.environ["CUDA_VISIBLE_DEVICES"] = '0' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    args['device'] = device
    print("device = {}".format(device))

    # Define and Load the model
    abs_sum = AbstractiveSummariser(args, device)
    trained_model = args['model_save_dir']+"abssum-{}-ep{}-bn{}.pt".format(args['model_name'],args['model_epoch'],args['model_bn'])
    abs_sum.load_state_dict(torch.load(trained_model))
    abs_sum.eval() # switch it to eval mode
    abs_sum.is_training = False
    print("Restored model from {}".format(trained_model))

    # Load and prepare data
    test_data = load_data(args, 'test')
    print("========================================================")
    print("start decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))
    print("========================================================")

    if args['decoding_method'] == 'greedysearch':
        with torch.no_grad():
            print("----------------- GREEDY SEARCH -----------------")
            greedy_search(abs_sum, test_data, args, start_idx, batch_size, num_batches)
    elif args['decoding_method'] == 'beamsearch':
        with torch.no_grad():
            print("------------------ BEAM SEARCH ------------------")
            beam_width = 5
            print("beam_width = {}".beam_width)
            beam_search(abs_sum, test_data, args, start_idx, batch_size, num_batches, k=beam_width)
    else:
        raise RuntimeError('decoding method not supported')

    print("finish decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python decode.py start_idx")
        raise Exception("argv error")

    start_idx = int(sys.argv[1])
    decode(start_idx)
