"""Inference time script for the abstractive task"""
import os
import sys
import torch
import pdb
from transformers import BertTokenizer

from models.abstractive import *
from train import load_data
from train_abstractive import get_a_batch_abs
from data2 import ProcessedDocument

if torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else: raise Exception("Torch Version not supoorted")

START_TOKEN = '[CLS]'
SEP_TOKEN   = '[SEP]'
STOP_TOKEN  = '[MASK]'

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

START_TOKEN_ID = bert_tokenizer.convert_tokens_to_ids(START_TOKEN)
SEP_TOKEN   = bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
STOP_TOKEN  = bert_tokenizer.convert_tokens_to_ids(STOP_TOKEN)

def greedy_search(model, data, args):
    # model = trained PyTorch abstractive model
    # data  = retrieved from load_data
    device = args['device']
    batch_size = args['batch_size']
    max_summary_length = args['max_summary_length']

    num_batches = int(data['num_data']/args['batch_size']) + 1
    time_step = args['max_summary_length']
    idx = 0

    for bn in range(num_batches):
        # check if it is the last batch
        if bn == (num_batches - 1): last_batch = True
        else: last_batch = False

        batch = get_a_batch_abs(
            data['encoded_articles'], data['attention_masks'],
            data['token_type_ids_arr'], data['cls_pos_arr'],
            data['target_pos'], args['max_num_sentences'],
            None, None, None, idx, batch_size,
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

        # print(bert_tokenizer.decode(tgt_ids[0].cpu().numpy()))
        # pdb.set_trace()
        idx += batch_size
        print("batch {}/{} finished".format(bn, num_batches))

def decode():
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['max_summary_length'] = 96
    args['batch_size'] = 64
    args['use_gpu'] = True
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "ANOV17A"
    args['model_epoch'] = 0
    args['model_bn'] = 6000
    args['decoding_method'] = 'greedysearch'
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
    print("Restored model from {}".format(trained_model))

    # Load and prepare data
    test_data = load_data(args, 'test')

    if args['decoding_method'] == 'greedysearch':
        with torch.no_grad():
            greedy_search(abs_sum, test_data, args)
    else:
        raise RuntimeError('decoding method not supported')

if __name__ == "__main__":
    decode()
