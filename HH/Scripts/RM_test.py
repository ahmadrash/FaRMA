import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset, load_from_disk
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import set_seed
import json

LLM_calls = 0
RM_calls = 0

def evaluate_reward_v1(Q, A, reward_tokenizer, reward_model):
    inputs = reward_tokenizer(Q, A, return_tensors='pt').to(device2)
    external_reward = reward_model(**inputs).logits[0].cpu().detach().item()
    return external_reward

def RGTG_decoding_v2(llm_model=None, llm_tokenizer=None, reward_model=None, reward_tokenizer=None, 
                topk=10, prompt=None, max_generation_length=128, mode=2, w=1.0):
    
    tokenizer_output = llm_tokenizer(prompt, return_tensors='pt').to(device1)
    input_ids = tokenizer_output.input_ids
    
    # use sequence to keep the entire generation
    sequence = torch.tensor([[]],dtype=torch.int64).to(device1)

    for t in range(0, max_generation_length):
        if t == 0:
            llm_output = llm_model(input_ids=input_ids)
            RM_output = reward_model(input_ids=input_ids.to(device2))
        else:
            llm_output = llm_model(input_ids=torch.cat((input_ids, sequence), dim=-1))
            RM_output = reward_model(torch.cat((input_ids, sequence), dim=-1).to(device2))
        
        global LLM_calls
        global RM_calls
        
        LLM_calls += 1
        RM_calls += 1
        
        LLM_logits = llm_output.logits[0, -1]
        RM_logits = RM_output.logits[0, -1].to(device1)
        RG_logit = LLM_logits[:RM_logits.shape[0]] + w * RM_logits
        
        if mode == 1:
            sampled_token_id = torch.topk(RG_logit, 1).indices[0].reshape(1, 1)
        elif mode == 2:
            topk_values, topk_indices = torch.topk(RG_logit, topk, dim=-1)
            sampled_index = torch.distributions.categorical.Categorical(logits=topk_values).sample().item()
            sampled_token_id = topk_indices[sampled_index].reshape(1, 1)
            
        sequence = torch.cat((sequence, sampled_token_id), dim=-1)
        
        if sequence[0][-1].item() == llm_tokenizer.eos_token_id:
            print(f"EOS BREAK: {t}")
            break
    
    generation = llm_tokenizer.decode(sequence[0])
    return {"sequence": generation}


def test_RGTG(prompt=None, topk=10, max_generation_length=128, mode=2, w=1.0):
    v2_score = 0
    # set_seed(43)
    
    v2_output = RGTG_decoding_v2(llm_model=base_model, llm_tokenizer=base_tokenizer, 
                reward_model=v2_reward_model, reward_tokenizer=v2_reward_tokenizer,
                topk=topk, prompt=prompt, max_generation_length=max_generation_length, mode=mode, w=w)
    v2_score = evaluate_reward_v1(prompt, v2_output['sequence'], v1_reward_tokenizer, v1_reward_model)
    print(f"v2_generation: {v2_output['sequence']}")
    print(f"v2_score: {round(v2_score, 4)}")
    print("\n")
    text_output.append({"prompt": prompt, "result": v2_output['sequence']})
    
    ret = {"v2_score": round(v2_score, 4)} 
    
    return ret

def test_all(sample_size=50, seed=42, topk=10, max_generation_length=128):
    HH_dataset = load_from_disk("./HH/Data/hh_SFT_eval")
    test_data = HH_dataset.shuffle(seed=seed)
    
    all_results = pd.DataFrame()

    for i in tqdm(range(0, sample_size)):
        prompt = f"{test_data[i]['prompt']}"
        print(f"Prompt:{prompt}\n")
        result = test_RGTG(prompt, topk, max_generation_length)
        all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)
        
    statistics_df = pd.DataFrame({
        'Mean': np.mean(all_results, axis=0),
        'Ste': np.std(all_results, axis=0) / np.sqrt(len(all_results))
    })
        
    print(f"SCORES:")
    print(statistics_df)
    
    with open("./HH/Results/v2.json", "w") as outfile:
        outfile.truncate()
        json.dump(text_output, outfile, ensure_ascii=False)

if __name__ == "__main__":
    device1 = "cuda:0"
    device2 = "cuda:1"
    
    text_output = []
    
    # base model
    base_tokenizer = AutoTokenizer.from_pretrained("lomahony/eleuther-pythia2.8b-hh-sft")
    base_model = AutoModelForCausalLM.from_pretrained("lomahony/eleuther-pythia2.8b-hh-sft", torch_dtype=torch.float16).to(device1)
    base_model.eval()
    
    # v1 reward models
    v1_reward_tokenizer = AutoTokenizer.from_pretrained("./HH/Models/RM_v1")
    v1_reward_model = AutoModelForSequenceClassification.from_pretrained("./HH/Models/RM_v1", num_labels=1, torch_dtype=torch.float16, cache_dir="./HH/Models/").to(device2)
    v1_reward_model.eval()
    
    # v2 reward models
    v2_reward_tokenizer = AutoTokenizer.from_pretrained("./HH/Models/RM_BT_iter7")
    v2_reward_model = AutoModelForCausalLM.from_pretrained("./HH/Models/RM_BT_iter7", torch_dtype=torch.float16, cache_dir="./HH/Models").to(device2)
    v2_reward_model.eval()

    # test all 
    test_all(sample_size=50, seed=44, topk=10, max_generation_length=128)
    print(f"LLM_calls: {LLM_calls}")
    print(f"RM_calls: {RM_calls}")
    
