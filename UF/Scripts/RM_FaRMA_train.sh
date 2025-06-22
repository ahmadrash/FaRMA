#!/bin/bash

START=1  
END=7

for ((i=START; i<=END; i++)); do
    prev_iter=$((i - 1))

    if [[ $prev_iter -eq 0 ]]; then
        model_name_or_path="alignment-handbook/zephyr-7b-sft-full"
    else
        model_name_or_path="./UF/Models/RM_Max_iter${prev_iter}"
    fi

    accelerate launch --config_file=./UF/default_config.yaml --num_processes 2 \
        ./UF/Scripts/RM_BT.py \
        --model_name_or_path="${model_name_or_path}" \
        --output_dir="./UF/Models/RM_BT_iter${i}" \
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

    accelerate launch --config_file=./UF/default_config.yaml --num_processes 2 \
        ./UF/Scripts/RM_Max.py  \
        --model_name_or_path="./UF/Models/RM_BT_iter${i}" \
        --output_dir="./UF/Models/RM_Max_iter${i}" \
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
        --save_strategy="no" \
        
done