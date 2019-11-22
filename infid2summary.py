"""to generate summaries after the inference step"""
import os
import sys
import pdb
from tqdm import tqdm
from nltk import tokenize

from data2 import load_data_pickle
from data2 import EOS_TOKENS


def generate_summary(filepath, output, top_k=3, trigram_blocking=False):
    documents = load_data_pickle(filepath)

    assert len(documents) == len(output), "len(documents) != len(output)"

    num_data = len(documents)

    summaries = [None for _ in range(num_data)]
    # IMPORTANT: to detokenize please refer to data2.py
    for idx, document in tqdm(enumerate(documents)):
        article = document[0].decode('utf-8')
        sentences = tokenize.sent_tokenize(article)

        S = []
        length = len(sentences)

        if trigram_blocking:
            # Trigram blocking
            # Given summary S and candidate sentence c
            # do not choose c if there is a trigram in c that is also in S
            seen_trigram = []
            for j in output[idx]:
                if j < length:
                    candidate = sentences[j]
                    words = tokenize.word_tokenize(candidate)
                    found_existing_tg = False
                else:
                    continue

                for p in range(len(words)-2):
                    tg = (words[p], words[p+1], words[p+2])
                    if tg not in seen_trigram:
                        seen_trigram.append(tg)
                    else:
                        found_existing_tg = True

                if found_existing_tg == False: S.append(candidate)
                else: pass

                if len(S) == top_k: break

        else:
            for j in output[idx][:top_k]:
                if j < length: S.append(sentences[j])

        if len(S) < 1:
            # pdb.set_trace()
            print("id {} len(s) == 0".format(idx))
        summaries[idx] = S

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
    ensemble = True
    if not ensemble:
        model = 'NOV15C'
        epoch = 1
        bn = 26000

        test_data_path = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/test.pk.bin'
        inf_path = '/home/alta/summary/pm574/summariser0/out_inference/extractive/model-{}-ep{}-bn{}.top10.txt'.format(model, epoch, bn)
        summary_out_dir = '/home/alta/summary/pm574/summariser0/out_summary/extractive/model-{}-ep{}-bn{}-tg/'.format(model, epoch, bn)
        output = read_inference_output(inf_path)
        summaries = generate_summary(test_data_path, output, top_k=3, trigram_blocking=True)
        write_summary_files(summary_out_dir, summaries)

    else:
        ensemble_name = "ensemble-NOV13Fc-NOV13G-NOV15B-NOV15C.avg"
        test_data_path = '/home/alta/summary/pm574/data/cnn_dm/finished_files_pm574/test.pk.bin'
        inf_path = '/home/alta/summary/pm574/summariser0/out_inference/extractive/{}.top10.txt'.format(ensemble_name)
        summary_out_dir = '/home/alta/summary/pm574/summariser0/out_summary/extractive/{}-tg/'.format(ensemble_name)
        output = read_inference_output(inf_path)
        summaries = generate_summary(test_data_path, output, top_k=3, trigram_blocking=True)
        write_summary_files(summary_out_dir, summaries)

if __name__ == "__main__":
    main()
