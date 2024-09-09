import os

from transformers import BertTokenizer
import torch
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


def calculate_entropy():
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # train_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train'])[0]
    f = open('/home/DiffuSeq/datasets/QQP/train.jsonl')
    train_data = pd.read_json(path_or_buf=f, lines=True)
    print(train_data)

    word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)
    count_length = defaultdict(int)

    # for data in tqdm(train_data['src']):
    #     for iid in (iids := tokenizer(data)['input_ids']):
    #         word_freq[iid] += 1

    for data in tqdm(train_data['trg']):
        for iid in (iids := tokenizer(data)['input_ids']):
            word_freq[iid] += 1
        count_length[len(iids)] += 1

    if not os.path.exists('./word_freq'):
        os.mkdir('word_freq')

    print(count_length)
    for i in sorted(list(count_length.items())):
        print(i)
    # torch.save(word_freq, f'./word_freq/bert-base-uncased_qqp.pt')

def calculate_tf():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    f = open('/home/DiffuSeq/datasets/QQP/train.jsonl')
    train_data = pd.read_json(path_or_buf=f, lines=True)
    print(train_data)

    word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)

    for data in tqdm(train_data['trg']):
        iids = set(tokenizer(data)['input_ids'])
        for iid in iids:
            word_freq[iid] += 1

    if not os.path.exists('./word_freq'):
        os.mkdir('word_freq')
    torch.save(word_freq, f'./word_freq/bert-base-uncased_qqp_tf.pt')


if __name__ == '__main__':
    calculate_tf()
    # word_freq = torch.load(f'../word_freq/bert-base-uncased_qqp.pt')
    # print(sum(word_freq))



