# FaRMA

We present the detailed steps required to train a reward model using the FaRMA method described in the "Towards Cost-Effective Reward Guided Text Generation" paper. Please refer to our paper [here](https://arxiv.org/abs/2502.04517) for more details.

Please cite our paper if you use this code in your work.

```
@inproceedings{rashid2025towards,
  title={Towards Cost-Effective Reward Guided Text Generation},
  author={Rashid, Ahmad and Wu,Ruotian, and Fan, Rongqi and Li, Hongliang and  Kristiadi, Agustinus and Poupart, Pascal},
  booktitle={ICML},
  year={2025}
}
```

## Overview:

1. Data preprocessing
2. Train reward model
3. Run evaluation tests

## Setup
 The code is tested with Python 3.10 and and CUDA 12.1

Install all the packages

```bash
cd FaRMA
pip install torch==2.6
pip install -r requirements.txt
```

Please note that we install torch before the rest of the packages due to a compatibality issue with the xformers package.

After installing the packages, the next step is to add the trainer class for the new architecture into the trl source files. 
```bash
cp FaRMA_trainer/reward_trainer_explicit.py {your_path_to_trl}/trainer
cp FaRMA_trainer/reward_trainer_optimal.py {your_path_to_trl}/trainer
```

Next, we will show the demonstration on how to replicate the training and inference for the TLDR dataset, but all the files for all the datasets are available and you can follow the same process to replicate the results of the other datasets.

## Step 1: Data Preprocessing
Run the following command to process the HH dataset. The created datasets are in `./TLDR/Dataset`

We need to create two datasets, one is the full-sequence preference dataset where we append the responses to the prompt to create a dataset contains the "chosen" and "rejected" only. The second dataset is for the purpose of the contraint loss, we create a dataset that contains only partial sequences of the "chosen" responses of each sample.

To run our evaluation script, you also need to create a SFT dataset which can be used for SFT training as well as the test dataset.

```bash
python ./TLDR/Scripts/TLDR_dataloader_pref_full.py
python ./TLDR/Scripts/TLDR_dataloader_pref_partial.py
python ./TLDR/Scripts/TLDR_dataloader_SFT.py
```

## Step 2: Train Reward Model
Run the following command to train the standard reward model on the Bradley-Terry loss only.

```bash
bash ./TLDR/Scripts/RM_train.sh
```

Run the following command to train the reward model by alternating between the Bradley-Terry loss and maximum constraint loss. The Model checkpoints are saved at `./TLDR/Models`

```bash
bash ./TLDR/Scripts/RM_FaRMA_train.sh
```

## Step 3.1: Run Evaluation Tests
Run the following command to get the generation results and their reward scores based on the full-sequence reward models. The generated sentences are stored in .json files.

```bash
python ./TLDR/Scripts/RM_test.py
```

## Step 3.2: test with vllm
We also provide the script to run evaluation with vllm. First, you need to comment out line 59 - 61 in the adapter file in your vllm path, to allow access to lm_head of the reward model. The path of the adapter file is `{your_path_to_vllm}/model_executor/models/adapters.py`. Then run the following commnad to get generation results and reward scores by vllm. This could reduce the generation time by approximately 50%.

```bash
export VLLM_USE_V1=0
python ./TLDR/Scripts/vllm_test.py
```

