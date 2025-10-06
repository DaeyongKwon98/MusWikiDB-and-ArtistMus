from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from string import Template
import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score
from collections import Counter
import argparse
import os
import random
import numpy as np

def main(args):
    # Set seed for reproducibility
    set_seed(42)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    
    # Load benchmark data
    df = pd.read_csv(args.benchmark_file)
    
    # Load retrieved documents if using RAG or rerank mode
    if args.mode in ["rag", "rerank"]:
        if not args.topk_file:
            raise ValueError("topk file is required for RAG and rerank modes")
        
        with open(args.topk_file, 'r', encoding='utf-8') as f:
            topk_data = json.load(f)
        
        df['retrieved_documents'] = topk_data
        df['topk'] = df['retrieved_documents'].apply(lambda x: x[:4] if isinstance(x, list) else [])
    
    # Run evaluation
    print(f"Evaluating benchmark dataset in {args.mode} mode...")
    accuracy = evaluate(model, tokenizer, df, "benchmark", args.mode)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"BENCHMARK accuracy: {accuracy:.3f} (mode: {args.mode})")

def set_seed(seed):
    """Set random seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(model_id):
    """Load ChatMusician model and tokenizer"""
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.model_max_length = 2048
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        resume_download=True
    ).eval()
    
    return model, tokenizer

def generate_answers(model, tokenizer, prompts):
    """Generate answers for given prompts using ChatMusician model"""
    prompt_template = Template("Human: ${inst}")
    
    generation_config = GenerationConfig(
        min_new_tokens=1,
        max_new_tokens=20,
        do_sample=False,
    )
    
    response_list = []
    
    for prompt in tqdm(prompts, desc="Generating answers"):
        formatted_prompt = prompt_template.safe_substitute({"inst": prompt})
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        
        try:
            with torch.no_grad():
                response = model.generate(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs['attention_mask'].to(model.device),
                    eos_token_id=tokenizer.eos_token_id,
                    generation_config=generation_config,
                )
            
            decoded_response = tokenizer.decode(
                response[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            if len(decoded_response) > 0:
                response_list.append(decoded_response[0].upper())
            else:
                response_list.append('X')
                
        except torch.cuda.OutOfMemoryError:
            print("OOM error! Skipping prompt.")
            torch.cuda.empty_cache()
            response_list.append('X')
            continue
    
    return response_list

def create_prompts(df, mode):
    """Create prompts based on the inference mode"""
    prompts = []
    
    if mode in ["rag", "rerank"]:
        # RAG/Rerank mode: use context from retrieved documents
        df['context'] = df['topk'].apply(lambda x: '\n\n'.join(x))
        
        for _, row in df.iterrows():
            prompt = f"Context: {row['context']}\n\nQuestion: {row['question']}\n\nAnswer:"
            prompts.append(prompt)
    
    else:  # zero-shot mode
        for _, row in df.iterrows():
            prompt = f"Question: {row['question']}\n\nAnswer:"
            prompts.append(prompt)
    
    return prompts

def evaluate(model, tokenizer, df, name, mode):
    """Evaluate model performance on the given dataset"""
    prompts = create_prompts(df, mode)
    answers = generate_answers(model, tokenizer, prompts)
    
    # Compare with ground truth
    gt = df['answer'].tolist()
    accuracy = accuracy_score(gt, answers)
    
    print(f"\n{name.upper()} accuracy: {accuracy:.3f}")
    print("Answer distribution:", Counter(answers))
    return round(accuracy, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatMusician inference for music understanding QA")
    parser.add_argument("--model_id", type=str, default="m-a-p/ChatMusician", help="HuggingFace ChatMusician model ID")
    parser.add_argument("--benchmark_file", type=str, required=True, help="CSV file containing benchmark questions")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "rag", "rerank"], required=True, help="Inference mode: zero-shot, rag, or rerank")
    parser.add_argument("--topk_file", type=str, help="JSON file containing retrieved documents (required for RAG/rerank)")   
    args = parser.parse_args()
    
    main(args)