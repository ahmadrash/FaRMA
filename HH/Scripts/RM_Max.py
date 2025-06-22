import warnings

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
import wandb
import numpy as np

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from trl.trainer.reward_trainer_optimal import RewardTrainerOptimal

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    # Align padding tokens between tokenizer and model
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    ##############
    # Load dataset
    ##############
    raw_datasets_train = load_from_disk('./HH/Data/hh_pref_train_partial')
    raw_datasets_eval = load_from_disk('./HH/Data/hh_pref_eval_partial')
    idxs = np.random.choice(len(raw_datasets_eval), size=64, replace=False)
    eval_dataset = raw_datasets_eval.select(idxs)
    idxs = np.random.choice(len(raw_datasets_train), size=6000, replace=False)
    train_dataset = raw_datasets_train.select(idxs)

    ##########
    # Training
    ##########
    trainer = RewardTrainerOptimal(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        # peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
