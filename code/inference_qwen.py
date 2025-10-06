import os
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import Counter
import json
import argparse

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
        
        df['retrieved_docs'] = topk_data
        
        # Limit number of chunks based on context window
        df['topk'] = df['retrieved_docs'].apply(lambda x: x[:4] if isinstance(x, list) else [])
    
    # Run evaluation
    print(f"Evaluating benchmark dataset in {args.mode} mode...")
    accuracy = evaluate(model, tokenizer, df, "benchmark", args.mode, args.batch_size, args.max_new_tokens, args.enable_thinking)
    
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
    """Load Qwen model and tokenizer with memory optimization"""
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    
    # Set pad_token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_answers(model, tokenizer, prompts, batch_size, max_new_tokens, enable_thinking=False):
    """Generate answers for given prompts using Qwen model with chat template"""
    generated_answers = []

    for i in tqdm(range(0, len(prompts), batch_size),
                  total=(len(prompts) + batch_size - 1) // batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = []

        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tokenize=False,
            )

            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Use deterministic output
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Extract generated tokens (excluding input length)
                output_ids = outputs[0][inputs["input_ids"].shape[-1]:].tolist()

                # Find thinking token id 151668 position
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                batch_outputs.append(content.lower().strip())

                # Memory cleanup
                del outputs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"OOM error! Skipping prompt.")
                torch.cuda.empty_cache()
                batch_outputs.append("ERROR")
                continue

        # Extract answers
        for output in batch_outputs:
            if 'answer: a' in output:
                generated_answers.append('A')
            elif 'answer: b' in output:
                generated_answers.append('B')
            elif 'answer: c' in output:
                generated_answers.append('C')
            elif 'answer: d' in output:
                generated_answers.append('D')
            elif output and output[0].lower() in ['a', 'b', 'c', 'd']:
                generated_answers.append(output[0].upper())
            else:
                print("Error in response:", output)
                generated_answers.append('X')
    
    return generated_answers

def create_prompts(df, mode):
    """Create prompts based on the inference mode"""
    prompts = []
    
    if mode in ["rag", "rerank"]:
        # RAG/Rerank mode: use context from retrieved documents
        df['context'] = df['topk'].apply(lambda x: '\n\n'.join(x))
        
        for _, row in df.iterrows():
            prompt = f"""### INSTRUCTION ###
You are an expert Q&A system. Your task is to answer the multiple-choice question below based ONLY on the provided context.
Read the context, question, and options carefully.
Your response MUST be in the format "Answer: [LETTER]", where [LETTER] is the single capital letter of the correct option.
Do not use any prior knowledge. Do not provide explanations or any other text.

### CONTEXT ###
{row['context']}

### QUESTION AND OPTIONS ###
Question: {row['question']}

### FINAL ANSWER ###
Answer:
""".strip()
            prompts.append(prompt)
    
    else:  # zero-shot mode
        for _, row in df.iterrows():
            prompt = f"""### INSTRUCTION ###
You are an automated evaluation system. Your sole task is to solve the multiple-choice question below.
Read the question and the provided options carefully.
Your response MUST be in the format "Answer: [LETTER]", where [LETTER] is the single capital letter of the correct option.
Do not write the text of the answer. Do not write any explanations. Do not complete the sentence in the question.

### QUESTION AND OPTIONS ###
Question: {row['question']}

### FINAL ANSWER ###
Answer:
""".strip()
            prompts.append(prompt)
    
    return prompts

def evaluate(model, tokenizer, df, name, mode, batch_size, max_new_tokens, enable_thinking=False):
    """Evaluate model performance on the given dataset"""
    prompts = create_prompts(df, mode)
    answers = generate_answers(model, tokenizer, prompts, batch_size, max_new_tokens, enable_thinking)

    # Compare with ground truth
    gt = df['answer'].tolist()
    accuracy = accuracy_score(gt, answers)

    print(f"\n{name.upper()} accuracy: {accuracy:.3f}")
    print("Answer distribution:", Counter(answers))
    return round(accuracy, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen LLM inference for music understanding QA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B", help="HuggingFace Qwen model ID")
    parser.add_argument("--benchmark_file", type=str, required=True, help="CSV file containing benchmark questions")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "rag", "rerank"], required=True, help="Inference mode: zero-shot, rag, or rerank")
    parser.add_argument("--topk_file", type=str, help="JSON file containing retrieved documents (required for RAG/rerank)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode for Qwen model")
    
    args = parser.parse_args()
    
    main(args)
