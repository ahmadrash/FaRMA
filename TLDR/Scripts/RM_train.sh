#!/bin/bash

accelerate launch --config_file=./default_config.yaml --num_processes 2 \
    ./TLDR/Scripts/RMv1.py \
    --model_name_or_path="meta-llama/Llama-3.2-1B-Instruct" \
    --output_dir="./TLDR/Models/RM_v1" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=5e-6 \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --max_length=512 \
    --bf16=True \
    --dataset-name="not needed" \
    --save_strategy="no"
