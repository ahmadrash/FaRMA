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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    path = f'./TLDR/Data'
    if not os.path.exists(path):
        os.makedirs(path)
        
    print("no cached dataset, downloading")
    raw_datasets = load_dataset('openai/summarize_from_feedback', 'comparisons')
    raw_train_dataset = raw_datasets['train']
    raw_eval_dataset = raw_datasets['validation']

    transformed_rows_train = []
    for row in raw_train_dataset:
        prompt = row['info']['post']
        if prompt == None:
            continue
        summaries = row['summaries']
        choice = int(row['choice'])
        transformed_rows_train.append({
            'chosen': f'Content: {prompt}\nSummary: {summaries[choice]["text"]}',
            'rejected': f'Content: {prompt}\nSummary: {summaries[1 - choice]["text"]}',
        })
    transformed_rows_eval = []
    for row in raw_eval_dataset:
        prompt = row['info']['post']
        if prompt == None:
            continue
        summaries = row['summaries']
        choice = int(row['choice'])
        transformed_rows_eval.append({
            'chosen': f'Content: {prompt}\nSummary: {summaries[choice]["text"]}',
            'rejected': f'Content: {prompt}\nSummary: {summaries[1 - choice]["text"]}',
        })
    raw_datasets_train = Dataset.from_list(transformed_rows_train)
    raw_datasets_eval = Dataset.from_list(transformed_rows_eval)

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        
        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets_train = raw_datasets_train.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    raw_datasets_eval = raw_datasets_eval.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    
    train_dataset = raw_datasets_train
    eval_dataset = raw_datasets_eval
    print("LENGTH BEFORE FILTERING:", len(train_dataset), len(eval_dataset))
    
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )
    print("LENGTH AFTER FILTERING:", len(train_dataset), len(eval_dataset))

    print_data(train_dataset, size=5)
    
    # Cache
    train_dataset.save_to_disk(f'{path}/TLDR_pref_full_train_llama')
    eval_dataset.save_to_disk(f'{path}/TLDR_pref_full_eval_llama')