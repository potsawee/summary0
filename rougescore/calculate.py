# pyrouge = ROUGE summarization evaluation package
# 1. convert summaries into a format ROUGE understands
# 2. automatically generate the ROUGE config file

# format of the summary directories
# All summaries should contain one sentence per line.

import sys
from pyrouge import Rouge155
import pdb

def compute_rouge(system_dir, model_dir):
    """
    system_dir: directory containing generated summaries
        file.001.txt
        file.002.txt
        file.003.txt
    model_dir:  directory containing (single) reference summaries
        file.001.txt
        file.002.txt
        file.003.txt

    to install pyrouge:
        1) pip install pyrouge
        2) pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
    more information: https://pypi.org/project/pyrouge/
    """

    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = model_dir

    r.system_filename_pattern = 'file.(\d+).txt'
    r.model_filename_pattern = 'file.#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    # output_dict = r.output_to_dict(output)

def main():
    if len(sys.argv) != 3:
        print('Usage: python3 calculate.py system_dir model_dir')
        return

    system_dir = sys.argv[1]
    model_dir  = sys.argv[2]

    compute_rouge(system_dir, model_dir)

if __name__ == "__main__":
    main()
