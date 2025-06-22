from vllm import LLM, SamplingParams
from vllm.config import PoolerConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset, load_from_disk
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

def evaluate_reward_v1(Q, A, reward_tokenizer, reward_model):
    inputs = reward_tokenizer(Q, A, return_tensors='pt').to(device)
    external_reward = reward_model(**inputs).logits[0].cpu().detach().item()
    return external_reward

# call to the RM to return the logits
def RM_call(model, token_ids, w=1.8):
    lm_head = RM.llm_engine.model_executor.driver_worker.model_runner.model.lm_head
    hidden_states = model.encode(prompt_token_ids=token_ids, use_tqdm=False)[0].outputs.data.to(lm_head.weight.device).to(lm_head.weight.dtype)
    logits = hidden_states @ lm_head.weight.T


    return w * logits

# LLM logit processor
def logit_proc(prompt_token_ids, generated_token_ids, logits_row):
    RM_logit = RM_call(RM, list(prompt_token_ids + generated_token_ids))
    return logits_row + RM_logit

def test_all(sample_size=50, seed=42, max_generation_length=128):
    tldr_dataset = load_dataset("CarperAI/openai_summarize_tldr")
    test_data = tldr_dataset["test"]
    test_data = test_data.shuffle(seed=seed)
    
    all_results = pd.DataFrame()
    sampling_params = SamplingParams(temperature=1.0, max_tokens=max_generation_length, logits_processors=[logit_proc])

    for i in tqdm(range(0, sample_size)):
        prompt = "Summarize: " + test_data[i]['prompt']
        v2_output = llm.generate(prompt, sampling_params)
        v2_output = v2_output[0].outputs[0].text
        text_output.append({"prompt": prompt, "result": v2_output})
        v2_score = evaluate_reward_v1(prompt, v2_output, v1_reward_tokenizer, v1_reward_model)
        result = {"v2_score": round(v2_score, 4)}
        all_results = pd.concat([all_results, pd.DataFrame([result])], ignore_index=True)
        
    statistics_df = pd.DataFrame({
        'Mean': np.mean(all_results, axis=0),
        'Ste': np.std(all_results, axis=0) / np.sqrt(len(all_results))
    })
        
    print(f"V2_SCORES:")
    print(statistics_df)
    
    with open("./TLDR/Results/v2.json", "w") as outfile:
        outfile.truncate()
        json.dump(text_output, outfile, ensure_ascii=False)



if __name__ == "__main__":
    text_output = []
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct",
             gpu_memory_utilization=0.30,
             max_model_len=1000,
             enable_prefix_caching=True,
             tensor_parallel_size=1,)
    RM = LLM(model="./TLDR/Models/RM_BT_iter2",
            gpu_memory_utilization=0.30,
            max_model_len=1000,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            task="reward", 
            override_pooler_config=PoolerConfig(pooling_type="LAST", softmax=False, normalize=False))
    
    device = "cuda:0"
    
    # v1 reward models
    v1_reward_tokenizer = AutoTokenizer.from_pretrained("./TLDR/Models/RM_v1", padding_side='left')
    v1_reward_model = AutoModelForSequenceClassification.from_pretrained("./TLDR/Models/RM_v1", num_labels=1, torch_dtype=torch.bfloat16).to(device)
    v1_reward_model.eval()
    
    # test all 
    test_all(sample_size=100, seed=44, max_generation_length=128)






