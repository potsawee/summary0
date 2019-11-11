"""to generate summaries after the inference step"""
import os
import sys
import pdb
from tqdm import tqdm
from nltk import tokenize

from data2 import load_data_pickle
from data2 import EOS_TOKENS


def generate_summary(filepath, output, top_k=3):
    documents = load_data_pickle(filepath)

    assert len(documents) == len(output), "len(documents) != len(output)"

    num_data = len(documents)

    summaries = [None for _ in range(num_data)]
    # IMPORTANT: to detokenize please refer to data2.py
    for idx, document in tqdm(enumerate(documents)):
        article = document[0].decode('utf-8')
        sentences = tokenize.sent_tokenize(article)

        s = []
        length = len(sentences)
        for j in output[idx][:top_k]:
            if j < length: s.append(sentences[j])

        if len(s) < 1:
            pdb.set_trace()
        summaries[idx] = s

    print("generate_summary done")
    return summaries

def read_inference_output(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    output = []
    for line in lines:
        indices = [int(x) for x in line.strip().split(':')[-1].split(',')]
        output.append(indices)

    return output

def write_summary_files(dir, summaries):
    if not os.path.exists(dir):
        os.makedirs(dir)

    num_data = len(summaries)
    for idx in tqdm(range(num_data)):
        filepath = dir + 'file.{}.txt'.format(idx)
        with open(filepath, 'w') as f:
            line = '\n'.join(summaries[idx])
            f.write(line)

    print("write summaries done!")
    return

def main():
    model = 'NOV7F'
    epoch = 0
    bn = 32000

    test_data_path = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/test.pk.bin'
    inf_path = '/home/alta/summary/pm574/summariser0/out_inference/extractive/model-{}-ep{}-bn{}.top5.txt'.format(model, epoch, bn)
    summary_out_dir = '/home/alta/summary/pm574/summariser0/out_summary/extractive/model-{}-ep{}-bn{}/'.format(model, epoch, bn)
    output = read_inference_output(inf_path)
    summaries = generate_summary(test_data_path, output)
    write_summary_files(summary_out_dir, summaries)

if __name__ == "__main__":
    main()
