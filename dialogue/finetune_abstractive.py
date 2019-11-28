import os
import sys
sys.path.append("/home/alta/summary/pm574/summariser0/")

import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle
import random
from datetime import datetime

from data_ami1 import load_data
from models.abstractive import *
from train_abstractive import get_a_batch_abs, adjust_lr_not_improved, adjust_lr2, shift_decoder_target, evaluate2, LabelSmoothingLoss

if   torch.__version__ == '1.1.0': KEYPADMASK_DTYPE = torch.uint8
elif torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else: raise Exception("Torch Version not supoorted")

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class DialogActLabeller(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(DialogActLabeller, self).__init__()
        self.device = device
        self.input_size  = input_size
        self.output_size = output_size
        self.linear  = nn.Linear(input_size, output_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # Initialisation
        for name, p in self.linear.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                if name[-4:] == 'bias': nn.init.zeros_(p)

        # Move to Deivce
        self.to(device)

    def forward(self, enc_output, max_num_sent, cls_pos, last_sep):
        # enc_output     = [batch_size, num_tokens, hidden_size]
        batch_size   = enc_output.shape[0]
        segment_vecs = torch.zeros((batch_size, max_num_sent, self.output_size)).to(self.device)
        x = self.linear(enc_output) # [batch_size, num_tokens, self.output_size]

        for i in range(batch_size):
            num_sentences = len(cls_pos[i])
            for j in range(num_sentences-1):
                idx1 = cls_pos[i][j]
                idx2 = cls_pos[i][j+1]
                segment_vecs[i,j,:] = torch.sum(x[i, idx1:idx2 ,:], dim=0, keepdim=False)

            if num_sentences > 1:
                idx1 = idx2
                idx2 = last_sep[i]+1
            else:
                idx1 = 0
                idx2 = last_sep[i]+1

            j = num_sentences-1

            if idx2 > idx1:
                segment_vecs[i,j,:] = torch.sum(x[i, idx1:idx2 ,:], dim=0, keepdim=False)
            elif idx2 == idx1:
                segment_vecs[i,j,:] = torch.sum(x[i, idx1: ,:], dim=0, keepdim=False)
            else:
                raise RuntimeError("something wrong #1")


        segment_vecs = self.logsoftmax(segment_vecs)

        return segment_vecs

def train_abstractive_model():
    print("Start training abstractive model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['max_pos_embed'] = 512
    args['max_num_sentences'] = 32
    args['update_nbatches'] = 5
    args['batch_size'] = 5
    args['num_epochs'] = 50
    args['val_batch_size'] = 64
    args['val_stop_training'] = 20
    args['random_seed'] = 28
    args['lr_enc'] = 1e-5
    args['lr_dec'] = 1e-5
    args['lr_sl']  = 1e-4
    args['adjust_lr'] = False
    # ---------------------------------------------------------------------------------- #
    args['use_gpu'] = True
    args['model_save_dir'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
    args['model_name'] = "DIALAMIXNOV28A"
    args['load_model'] = "/home/alta/summary/pm574/summariser0/lib/trained_models/abssum-ANOV21A-ep9-bn40000.pt"
    # args['load_model'] = None
    args['best_val_loss'] = 1e+10
    # ---------------------------------------------------------------------------------- #
    args['max_summary_length'] = 96
    args['label_smoothing'] = 0.1
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

    # Get the data
    print("========== Loading train data ==========")
    train_data_ = load_data('train', args['max_pos_embed'], args['max_summary_length'])
    print("========== Loading valid data ==========")
    val_data_   = load_data('valid', args['max_pos_embed'], args['max_summary_length'])

    # ----------------------- Prepare the data ----------------------- #
    processed_documents = train_data_['in']
    processed_summaries = train_data_['sum']
    train_dacts         = train_data_['da']
    N = len(processed_documents)
    N2 = len(processed_summaries)
    x1 = [None] * N
    x2 = [None] * N
    x3 = [None] * N
    x4 = [None] * N
    y1 = [None] * N
    y2 = [None] * N
    for j in range(N):
        x1[j] = processed_documents[j].encoded_article
        x2[j] = processed_documents[j].attention_mask
        x3[j] = processed_documents[j].token_type_ids
        x4[j] = processed_documents[j].cls_pos
        y1[j] = processed_summaries[j].encoded_abstract
        y2[j] = processed_summaries[j].length
    dummy_target_pos = [0] * N
    train_data = {'encoded_articles': x1, 'attention_masks': x2,
                  'token_type_ids_arr': x3, 'cls_pos_arr': x4,
                  'target_pos': dummy_target_pos, 'num_data': N}
    train_summary = {'encoded_abstracts': y1, 'abs_lengths': y2, 'num_data': N2}

    processed_documents = val_data_['in']
    processed_summaries = val_data_['sum']
    val_dacts         = val_data_['da']
    N = len(processed_documents)
    N2 = len(processed_summaries)
    x1 = [None] * N
    x2 = [None] * N
    x3 = [None] * N
    x4 = [None] * N
    y1 = [None] * N
    y2 = [None] * N
    for j in range(N):
        x1[j] = processed_documents[j].encoded_article
        x2[j] = processed_documents[j].attention_mask
        x3[j] = processed_documents[j].token_type_ids
        x4[j] = processed_documents[j].cls_pos
        y1[j] = processed_summaries[j].encoded_abstract
        y2[j] = processed_summaries[j].length
    dummy_target_pos = [0] * N
    val_data = {'encoded_articles': x1, 'attention_masks': x2,
                'token_type_ids_arr': x3, 'cls_pos_arr': x4,
                'target_pos': dummy_target_pos, 'num_data': N}
    val_summary = {'encoded_abstracts': y1, 'abs_lengths': y2, 'num_data': N2}
    # ------------------------------------------------------------------- #

    # random seed
    random.seed(args['random_seed'])

    abs_sum = AbstractiveSummariser(args, device=device)
    hidden_size = abs_sum.encoder.bert.model.config.hidden_size
    DA_CLASSES  = 15
    da_labeller = DialogActLabeller(hidden_size, DA_CLASSES, device)
    print(abs_sum)
    print(da_labeller)

    # Load model if specified (path to pytorch .pt)
    pretrained_dict = torch.load(args['load_model'])
    abs_sum.load_state_dict(pretrained_dict)
    abs_sum.train()
    print("Loaded model from {}".format(args['load_model']))

    # Hyperparameters
    BATCH_SIZE = args['batch_size']
    NUM_EPOCHS = args['num_epochs']
    VAL_BATCH_SIZE = args['val_batch_size']
    VAL_STOP_TRAINING = args['val_stop_training']

    vocab_size = abs_sum.decoder.linear_decoder.out_features

    if args['label_smoothing'] > 0.0:
        criterion = LabelSmoothingLoss(num_classes=vocab_size,
                        smoothing=args['label_smoothing'], reduction='none')
    else:
        criterion = nn.NLLLoss(reduction='none')

    criterion_da = nn.NLLLoss(reduction='none')


    # we use two separate optimisers (encoder & decoder)
    optimizer_enc = optim.Adam(abs_sum.encoder.parameters(),lr=args['lr_enc'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer_dec = optim.Adam(abs_sum.decoder.parameters(),lr=args['lr_dec'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer_sl  = optim.Adam(da_labeller.parameters(), lr=args['lr_sl'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer_enc.zero_grad()
    optimizer_dec.zero_grad()
    optimizer_sl.zero_grad()

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
        train_data, train_summary, train_dacts = shuffle_data3(train_data, train_summary, train_dacts)

        idx = 0

        for bn in range(num_batches):
            # adjust the learning rate of the optimizer
            if args['adjust_lr']:
                adjust_lr2(optimizer_enc, optimizer_dec,
                        epoch, epoch_size=num_batches, bn=bn,
                        warmup_enc=20000, warmup_dec=10000)

            # check if it is the last batch
            if bn == (num_batches - 1):
                last_batch = True
                continue # unepxected error --- num_data % batch_size = 0
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
            encoder_output, decoder_output = abs_sum.multitask_forward(batch)


            # ------------------------ Decoder Loss ------------------------ #
            loss = criterion(decoder_output.view(-1, vocab_size), decoder_target)
            loss = (loss * decoder_mask).sum() /  decoder_mask.sum()

            # ------------- Dialogue Acts - Multitask Learning ------------- #
            batch_cls_pos = batch['encoder'][0][-1] # len(batch_cls_pos) = batch_size
            batch_last_sep_pos = last_sep_position(batch['encoder'][0][0])
            max_num_sent  = get_max_num_sent(batch_cls_pos)
            segment_vecs  = da_labeller(encoder_output, max_num_sent, batch_cls_pos, batch_last_sep_pos)
            # get DA labels for this batch [idx, idx+batch_size)
            if not last_batch:
                da_labels, da_mask = get_da_labels(train_dacts, max_num_sent, idx, BATCH_SIZE, device)
            else:
                da_labels, da_mask = get_da_labels(train_dacts, max_num_sent, idx, train_data['num_data']-idx, device)
            da_labels = da_labels.view(-1)
            da_mask = da_mask.view(-1)
            loss_da = criterion_da(segment_vecs.view(-1, DA_CLASSES), da_labels)
            loss_da = (loss_da * da_mask).sum() / da_mask.sum()
            # -------------------------------------------------------------- #
            loss_all = loss + loss_da
            loss_all.backward()

            idx += BATCH_SIZE

            if bn % args['update_nbatches'] == 0:
                # gradient_clipping
                max_norm = 1.0
                nn.utils.clip_grad_norm_(abs_sum.parameters(), max_norm)
                nn.utils.clip_grad_norm_(da_labeller.parameters(), max_norm)
                # update the gradients
                optimizer_enc.step()
                optimizer_dec.step()
                optimizer_sl.step()
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
                optimizer_sl.zero_grad()

            if bn % 250 == 0:
                _tgt  = bert_tokenizer.decode(batch['decoder'][0][0].cpu().numpy())
                _pred = bert_tokenizer.decode(torch.argmax(decoder_output[0], dim=-1).cpu().numpy())
                _i = _tgt.find('[MASK]')
                print("TARGET:", _tgt[:_i])
                print("PREDIC:", _pred[:_i])

            if bn % 50 == 0:
                print("[{}] batch number {}/{}: loss = {:.6} & loss_da = {:.6}".format(str(datetime.now()), bn, num_batches, loss, loss_da))
                sys.stdout.flush()

            if bn == 0: # e.g. eval every epoch
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at epoch {} step {}".format(epoch, bn))
                print("learning_rate_encoder = {}".format(optimizer_enc.param_groups[0]['lr']))
                print("learning_rate_decoder = {}".format(optimizer_dec.param_groups[0]['lr']))
                print("learning_rate_labeller = {}".format(optimizer_sl.param_groups[0]['lr']))
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
                        adjust_lr_not_improved(optimizer_enc, optimizer_dec, factor=0.5)
                        stop_counter += 1

                    else:
                        print("Model has not improved for {} times! Stop training.".format(VAL_STOP_TRAINING))
                        return

    print("End of training abstractive model")

def shuffle_data3(data_dict, summary_dict, dialogueacts):
    assert data_dict['num_data'] == summary_dict['num_data'], \
        "data_dict['num_data'] != summary_dict['num_data']"

    assert data_dict['num_data'] == len(dialogueacts), \
        "data_dict['num_data'] != len(dialogueacts)"

    _x = list(zip(
        data_dict['encoded_articles'],
        data_dict['attention_masks'],
        data_dict['token_type_ids_arr'],
        data_dict['cls_pos_arr'],
        data_dict['target_pos'],
        summary_dict['encoded_abstracts'],
        summary_dict['abs_lengths'],
        dialogueacts
    ))

    random.shuffle(_x)
    x1, x2, x3, x4, x5, y1, y2, z1 = zip(*_x)

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
    shuffled_dialogueacts = z1

    return shuffled_data_dict, shuffled_summary_dict, shuffled_dialogueacts

def get_da_labels(dialogueacts, max_num_sent, idx, batch_size, device):
    da_labels = [None for _ in range(batch_size)]
    da_mask   = [None for _ in range(batch_size)]
    for j in range(batch_size):
        slen = len(dialogueacts[idx+j])
        if slen <= max_num_sent:
            da_mask[j]   = [1.0]*slen + [0.0]*(max_num_sent-slen)
            da_labels[j] = dialogueacts[idx+j] + [0]*(max_num_sent-slen)
        else:
            da_mask[j] = [1.0]*max_num_sent
            da_labels[j] = dialogueacts[idx+j][:max_num_sent]

    return torch.tensor(da_labels).to(device), torch.tensor(da_mask).to(device)

def get_max_num_sent(cls_pos):
    max_count = 0
    for cls in cls_pos:
        count = len(cls)
        if count > max_count:
            max_count = count
    return max_count

def last_sep_position(input_ids):
    SEP_TOKEN_ID = 102
    batch_size = input_ids.shape[0]
    length     = input_ids.shape[1]
    input_ids  = input_ids.cpu().numpy()

    last_sep_pos_arr = [None for _ in range(batch_size)]
    for i in range(batch_size):
        idx = -1
        while True:
            if input_ids[i][idx] == SEP_TOKEN_ID:
                break
            else:
                idx -= 1
        last_sep_pos_arr[i] = length + idx

    return last_sep_pos_arr

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # ----- EVALUATION ----- #
        if sys.argv[1] == 'eval':
            path_to_model = sys.argv[2]
            evaluate_model(path_to_model)
        else:
            raise Exception("do you want to do evaluation?")
    elif len(sys.argv) == 1:
        # ------ TRAINING ------ #
        train_abstractive_model()
