import os
import sys
import torch
# import torch.nn as nn
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
    args['model_name'] = "NOV13Fc"
    args['model_epoch'] = 0
    args['model_bn'] = 4000
    top_k = 10


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
    print("device = {}".format(device))

    # Define and Load the model
    ext_sum = ExtractiveSummariser(args, device)
    trained_model = args['model_save_dir']+"extsum-{}-ep{}-bn{}.pt".format(args['model_name'],args['model_epoch'],args['model_bn'])
    ext_sum.load_state_dict(torch.load(trained_model))
    ext_sum.eval() # switch it to eval mode
    print("Restored model from {}".format(trained_model))

    # Load and prepare data
    test_data = load_data(args, 'test')

    # Hyperparameters
    batch_size = 256

    # Binary Cross Entropy Loss for the Extractive Task
    criterion = nn.BCELoss(reduction='none')

    # test losses
    total_test_loss = 0.0
    total_test_sentences = 0

    num_batches = int(test_data['num_data'] / batch_size) + 1
    print("num_batches = {}".format(num_batches))
    idx = 0

    # To store the predictions
    summaries = [[-1] for _ in range(test_data['num_data'])]

    with torch.no_grad():
        for bn in range(num_batches):
            # check if it is the last batch
            if bn == (num_batches - 1): last_batch = True
            else: last_batch = False

            # get my data
            inputs, targets, key_padding_mask, ms = \
                get_a_batch(test_data['encoded_articles'], test_data['attention_masks'],
                            test_data['token_type_ids_arr'], test_data['cls_pos_arr'],
                            test_data['target_pos'], args['max_num_sentences'],
                            idx, batch_size, last_batch, device)
            mask = ms[0]
            lengths = ms[1]

            # forward + backward + optimize
            sent_scores = ext_sum(inputs, key_padding_mask)

            # compute loss --- may be useless but compute anyway
            loss = criterion(sent_scores, targets)
            total_test_loss += (loss * mask.float()).sum().item()
            total_test_sentences += lengths.sum().item()

            if device == 'cuda':
                sent_scores = sent_scores.cpu()

            if last_batch:
                batch_size = test_data['num_data'] - idx

            for j in range(batch_size):
                indices = sent_scores[j].data.numpy().argsort()[-top_k:][::-1]
                summaries[idx+j] = indices

            idx += batch_size

            print("[{}] batch number {}/{}".format(str(datetime.now()), bn, num_batches))
            sys.stdout.flush()


    avg_test_loss = total_test_loss / total_test_sentences
    print("\navg_test_loss_per_sentence = {}".format(avg_test_loss))

    output_name = "out_inference/extractive/model-{}-ep{}-bn{}.top{}.txt" \
                  .format(args['model_name'],args['model_epoch'],args['model_bn'],top_k)

    with open(output_name , 'w') as f:
        for i in range(test_data['num_data']):
            summary_ids = ",".join([str(x) for x in summaries[i]])
            f.write("doc{}:{}\n".format(i, summary_ids))
    print("wrote: {}".format(output_name))
    print("End of inference extractive model")

if __name__ == "__main__":
    inference_extractive_model()
