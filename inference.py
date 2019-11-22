"""Inference script for Extractive model"""
import os
import sys
import torch
import numpy as np
import pdb
import pickle
from datetime import datetime

from models.extractive import *
from data2 import ProcessedDocument

from train import load_data, get_a_batch, generate_sequence_mask

def inference_extractive_model():
    print("Start performing inference using a trained extractive model")

    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "NOV15C"
    args['model_epoch'] = 1
    args['model_bn'] = 26000
    args['batch_size'] = 256
    top_k = 10
    # --------------------------------------- Ensemble --------------------------------------- #
    ensemble        = True # Please specify a set of model below
    ensemble_name   = "ensemble-NOV13Fc-NOV13G-NOV15B-NOV15C"
    combine_method  = "norm"
    ensemble_models = [
        "/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-NOV13Fc-ep0-bn4000.pt",
        "/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-NOV13G-ep2-bn6000.pt",
        "/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-NOV15B-ep1-bn34000.pt",
        "/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-NOV15C-ep1-bn26000.pt",
    ]
    # ---------------------------------------------------------------------------------------- #
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

    # Load and prepare data
    test_data = load_data(args, 'test')

    if not ensemble:
        # Define and Load the model
        ext_sum = ExtractiveSummariser(args, device)
        trained_model = args['model_save_dir']+"extsum-{}-ep{}-bn{}.pt".format(args['model_name'],args['model_epoch'],args['model_bn'])
        ext_sum.load_state_dict(torch.load(trained_model))
        ext_sum.eval() # switch it to eval mode
        print("Restored model from {}".format(trained_model))

        sent_score_array = single_model_inference(ext_sum, test_data, args, device)

    else: # doing ensemble
        print("Ensemble Inference --- {} --- combine method = {}".format(ensemble_name, combine_method))
        sent_score_arrays = [None for _ in range(len(ensemble_models))]
        for i, trained_model in enumerate(ensemble_models):
            ext_sum = ExtractiveSummariser(args, device)
            ext_sum.load_state_dict(torch.load(trained_model))
            ext_sum.eval() # switch it to eval mode
            print("Restored model from {}".format(trained_model))

            score = single_model_inference(ext_sum, test_data, args, device)
            sent_score_arrays[i] = score

        ensemble_score = np.zeros((test_data['num_data'], args['max_num_sentences']))
        if combine_method == 'avg':
            # METHOD1: just simply doing summation
            for score in sent_score_arrays:
                ensemble_score += score
        elif combine_method == 'norm':
            # METHOD2: normalise aross all sentence in one doc
            # P(sent1)+P(sent2)+...P(sent32) = 1.0
            for score in sent_score_arrays:
                ensemble_score += score / score.sum(axis=1, keepdims=1)
        else:
            raise RuntimeError("ensemble combination method not supported")

        sent_score_array = ensemble_score

    summaries = [[-1] for _ in range(test_data['num_data'])]
    for j in range(sent_score_array.shape[0]):
        indices = sent_score_array[j].argsort()[-top_k:][::-1]
        summaries[j] = indices

    if not ensemble:
        output_name = "out_inference/extractive/model-{}-ep{}-bn{}.top{}.txt" \
                      .format(args['model_name'],args['model_epoch'],args['model_bn'],top_k)
    else:
        output_name = "out_inference/extractive/{}.{}.top{}.txt" \
                      .format(ensemble_name, combine_method, top_k)

    with open(output_name , 'w') as f:
        for i in range(test_data['num_data']):
            summary_ids = ",".join([str(x) for x in summaries[i]])
            f.write("doc{}:{}\n".format(i, summary_ids))
    print("wrote: {}".format(output_name))
    print("End of inference extractive model")

def single_model_inference(model, data, args, device):
    max_num_sentences = args['max_num_sentences']
    batch_size = args['batch_size']

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')

    # test losses
    total_test_loss = 0.0
    total_test_sentences = 0

    num_batches = int(data['num_data'] / batch_size) + 1
    print("num_batches = {}".format(num_batches))
    idx = 0

    # To store the predictions
    # summaries = [[-1] for _ in range(data['num_data'])]
    sent_score_array = np.zeros((data['num_data'], max_num_sentences))

    with torch.no_grad():
        for bn in range(num_batches):
            # check if it is the last batch
            if bn == (num_batches - 1): last_batch = True
            else: last_batch = False

            # get my data
            inputs, targets, key_padding_mask, ms = \
                get_a_batch(data['encoded_articles'], data['attention_masks'],
                            data['token_type_ids_arr'], data['cls_pos_arr'],
                            data['target_pos'], max_num_sentences,
                            idx, batch_size, last_batch, device)
            mask = ms[0]
            lengths = ms[1]

            # forward + backward + optimize
            sent_scores = model(inputs, key_padding_mask)

            # compute loss --- may be useless but compute anyway
            loss = criterion(sent_scores, targets)
            total_test_loss += (loss * mask.float()).sum().item()
            total_test_sentences += lengths.sum().item()

            if device == 'cuda':
                sent_scores = sent_scores.cpu()

            if last_batch:
                batch_size = data['num_data'] - idx

            sent_score_array[idx:idx+batch_size, :] = sent_scores.data.numpy()

            # for j in range(batch_size):
            #     indices = sent_scores[j].data.numpy().argsort()[-top_k:][::-1]
            #     summaries[idx+j] = indices

            idx += batch_size

            print("[{}] batch number {}/{}".format(str(datetime.now()), bn, num_batches))
            sys.stdout.flush()

    avg_test_loss = total_test_loss / total_test_sentences
    print("avg_test_loss_per_sentence = {}".format(avg_test_loss))

    # return summaries
    return sent_score_array

if __name__ == "__main__":
    inference_extractive_model()
