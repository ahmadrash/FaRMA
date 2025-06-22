import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, GPT2Tokenizer, AutoModelForCausalLM
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import os
import torch.utils.data as data_utils
from transformers import GPT2Tokenizer, DataCollatorWithPadding, DefaultDataCollator, DataCollatorForSeq2Seq, RobertaTokenizer
from datasets import load_dataset, load_from_disk, Dataset # huggingface datasets
from typing import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os

def print_data(dataset, size=10):
    for i, data in enumerate(dataset):
        if i >= size:
            break
        print(data)
        

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia410m-hh-sft")
    tokenizer.pad_token = tokenizer.eos_token

    path = f'./HH/Data'
    if not os.path.exists(path):
        os.makedirs(path)
        
    print("no cached dataset, downloading")
    raw_datasets = load_dataset('Dahoas/static-hh')
    idxs = np.random.choice(len(raw_datasets['train']), size=80000, replace=False)
    raw_train_dataset = raw_datasets['train'].select(idxs)
    idxs = np.random.choice(len(raw_datasets['test']), size=2048, replace=False)
    raw_eval_dataset = raw_datasets['test'].select(idxs)
    
    transformed_rows_train = []
    for row in raw_train_dataset:
        prompt = row['prompt']
        chosen = row['chosen']
        transformed_rows_train.append({
            'prompt': prompt,
            'label': chosen,
        })
        
    transformed_rows_eval = []
    for row in raw_eval_dataset:
        prompt = row['prompt']
        chosen = row['chosen']
        transformed_rows_eval.append({
            'prompt': prompt,
            'label': chosen,
        })
    
    raw_datasets_train = Dataset.from_list(transformed_rows_train)
    raw_datasets_eval = Dataset.from_list(transformed_rows_eval)

    print_data(raw_datasets_train, size=10)
    
    # Cache
    raw_datasets_train.save_to_disk(f'{path}/hh_SFT_train')
    raw_datasets_eval.save_to_disk(f'{path}/hh_SFT_eval')









