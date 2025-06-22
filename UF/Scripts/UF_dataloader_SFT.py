import warnings
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, GPT2Tokenizer, AutoModelForCausalLM
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
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
    
    tokenizer = AutoTokenizer.from_pretrained("alignment-handbook/zephyr-7b-sft-full")
    tokenizer.pad_token = tokenizer.eos_token
    path = f'./UF/Data'
    if not os.path.exists(path):
        os.makedirs(path)
        
    print("no cached dataset, downloading")
    raw_datasets = load_dataset('HuggingFaceH4/ultrafeedback_binarized')
    raw_train_dataset = raw_datasets['train_prefs']
    raw_eval_dataset = raw_datasets['test_prefs']

    # print_data(raw_train_dataset, size=10)
    
    transformed_rows_train = []
    for row in raw_train_dataset:
        prompt = row['prompt']
        chosen = row["chosen"][1]["content"]
        
        prompt_id_len = len(tokenizer(prompt)["input_ids"])
        chosen_id_len = len(tokenizer(chosen)["input_ids"])
        if prompt_id_len + chosen_id_len > 1024:
            continue
        
        transformed_rows_train.append({
            'prompt': prompt,
            'label': chosen,
        })
        
    transformed_rows_eval = []
    for row in raw_eval_dataset:
        prompt = row['prompt']
        chosen = row["chosen"][1]["content"]
        
        prompt_id_len = len(tokenizer(prompt)["input_ids"])
        chosen_id_len = len(tokenizer(chosen)["input_ids"])
        if prompt_id_len + chosen_id_len > 1024:
            continue
        
        transformed_rows_eval.append({
            'prompt': prompt,
            'label': chosen,
        })

    raw_datasets_train = Dataset.from_list(transformed_rows_train)
    raw_datasets_eval = Dataset.from_list(transformed_rows_eval)

    print_data(raw_datasets_train, size=10)
    
    # Cache
    raw_datasets_train.save_to_disk(f'{path}/UF_SFT_train')
    raw_datasets_eval.save_to_disk(f'{path}/UF_SFT_eval')









