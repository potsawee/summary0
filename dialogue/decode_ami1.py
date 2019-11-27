"""Inference time script for the abstractive task"""
import os
import sys
sys.path.append("/home/alta/summary/pm574/summariser0/")

import torch
import pdb
import numpy as np
from datetime import datetime

from transformers import BertTokenizer

from models.abstractive import *
# from train import load_data
from data_ami1 import load_data
from train_abstractive import get_a_batch_abs
from data2 import ProcessedDocument, ProcessedSummary
from decode import beam_search_v2

def decode_ami():
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['max_summary_length'] = 96
    args['use_gpu'] = True
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "DIALAMINOV27E"
    args['model_epoch'] = 2
    args['model_bn'] = 0
    args['decoding_method'] = 'beamsearch'
    # ---------------------------------------------------------------------------------- #
    start_idx   = 0
    batch_size  = 10
    num_batches = 40
    beam_width  = 1
    alpha       = 1.0
    # ---------------------------------------------------------------------------------- #
    args['summary_out_dir'] = \
    '/home/alta/summary/pm574/summariser0/dialogue/out_summary/goochen/model-{}-ep{}-bn{}-{}{}-alpha{}/' \
    .format(args['model_name'], args['model_epoch'], args['model_bn'], args['decoding_method'], beam_width, alpha)
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
    if device == 'cuda':
        abs_sum.load_state_dict(torch.load(trained_model))
    elif device == 'cpu':
        abs_sum.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))

    abs_sum.eval() # switch it to eval mode
    print("Restored model from {}".format(trained_model))

    # Load and prepare data
    test_data_ = load_data('test', args['max_pos_embed'], args['max_summary_length'])
    processed_documents = test_data_['in']
    N = len(processed_documents)
    x1 = [None] * N
    x2 = [None] * N
    x3 = [None] * N
    x4 = [None] * N
    for j in range(N):
        x1[j] = processed_documents[j].encoded_article
        x2[j] = processed_documents[j].attention_mask
        x3[j] = processed_documents[j].token_type_ids
        x4[j] = processed_documents[j].cls_pos
    dummy_target_pos = [0] * N
    test_data = {'encoded_articles': x1, 'attention_masks': x2,
                 'token_type_ids_arr': x3, 'cls_pos_arr': x4,
                 'target_pos': dummy_target_pos, 'num_data': N}


    print("========================================================")
    print("start decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))
    print("========================================================")

    if args['decoding_method'] == 'beamsearch':
        with torch.no_grad():
            print("------------------ BEAM SEARCH ------------------")
            print("beam_width = {}".format(beam_width))
            beam_search_v2(abs_sum, test_data, args, start_idx, batch_size, num_batches, k=beam_width, alpha=alpha)
    else:
        raise RuntimeError('decoding method not supported')

    print("finish decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))

if __name__ == "__main__":
    if(len(sys.argv) != 1):
        print("Usage: python decode_ami1.py")
        raise Exception("argv error")
    decode_ami()
