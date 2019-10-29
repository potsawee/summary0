import pdb
from models.extractive import *

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn

def main():
    print("Start Experiment.py...")
    device = 'cpu'
    args = None
    ext_sum = ExtractiveSummeriser(args, device)

    input_ids, attention_mask, token_type_ids, position_ids, cls_pos = create_dummy_input()
    inputs = (input_ids, attention_mask, token_type_ids, position_ids, cls_pos)
    sent_scores = ext_sum(inputs, cls_pos)

    pdb.set_trace()


    print("Finish Experiment.py...")

def create_dummy_input():
    batch_size = 50
    sequence_length = 512
    max_num_sentences = 100


    input_ids = torch.randint(low=0,high=20000, size=(batch_size, sequence_length))
    attention_mask = torch.randint(low=0,high=2, size=(batch_size, sequence_length))
    token_type_ids = torch.randint(low=0,high=2, size=(batch_size, sequence_length))
    position_ids = torch.randint(low=0,high=2, size=(batch_size, sequence_length))

    # cls_pos = np.random.randint(low=0,high=512, size=(batch_size,)).tolist()
    cls_pos = [None for x in range(batch_size)]
    for i in range(batch_size):
        n = np.random.randint(low=1, high=max_num_sentences)
        pos = [0]
        while True:
            new_pos = pos[-1] + np.random.randint(low=5, high=30)
            if new_pos < sequence_length:
                pos.append(new_pos)
            else:
                break
        cls_pos[i] = pos

    # cls_pos[20] = [0, 10, 20, 30, 40, 56, 60, 66, 71, 100, 120, 150, 200, 210, 220, 250, 300, 310, 320, 400, 450, 460]
    # pp = 0
    # for j in range(batch_size):
    #     if len(cls_pos[j]) > pp:
    #         pp = len(cls_pos[j])
    # print("max length=", pp)

    return (input_ids, attention_mask, token_type_ids, position_ids, cls_pos)


main()
