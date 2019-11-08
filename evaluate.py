import torch
import torch.nn as nn
import pdb
import os
import sys

from models.extractive import *
from data2 import ProcessedDocument
from train import load_data, get_a_batch, generate_sequence_mask

def evaluate_model(path_to_model, data_type):
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
            # pdb.set_trace()
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '3' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    # Define the model
    ext_sum = ExtractiveSummeriser(args, device)
    ext_sum.load_state_dict(torch.load(path_to_model))
    ext_sum.eval()

    # Load the data
    data = load_data(args, data_type)

    # Hyperparameters
    BATCH_SIZE = 256 # 256 works fine on GeForce GTX 1080 ti

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')

    num_batches = int(data['num_data'] / BATCH_SIZE) + 1
    print("num_batches = {}".format(num_batches))
    idx = 0

    total_eval_loss = 0.0
    total_eval_sentences = 0

    with torch.no_grad():
        for bn in range(num_batches):

            # check if it is the last batch
            if bn == (num_batches - 1): last_batch = True
            else: last_batch = False

            inputs, targets, ms = get_a_batch(data['encoded_articles'], data['attention_masks'],
                                            data['token_type_ids_arr'], data['cls_pos_arr'],
                                            data['target_pos'], args['max_num_sentences'],
                                            idx, BATCH_SIZE, last_batch, device)
            mask = ms[0]
            lengths = ms[1]

            # forward + backward + optimize
            sent_scores = ext_sum(inputs)


            loss = criterion(sent_scores, targets)
            this_loss = (loss * mask.float()).sum().data
            total_eval_loss += this_loss
            total_eval_sentences += lengths.sum().data

            idx += BATCH_SIZE

            print(this_loss)

            print("#", end="")
            sys.stdout.flush()

    print("")
    print("model = {}".format(path_to_model))
    avg_eval_loss_per_sentence = total_eval_loss / total_eval_sentences
    print("avg_eval_loss_per_sentence = {}".format(avg_eval_loss_per_sentence))

if __name__ == "__main__":
    path_to_model = '/home/alta/summary/pm574/summariser0/lib/trained_models/extsum-tval0-ep0-bn10000.pt'
    data_type = 'val'
    evaluate_model(path_to_model, data_type)
