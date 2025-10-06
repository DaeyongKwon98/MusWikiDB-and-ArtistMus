import pandas as pd
from tqdm import tqdm
from openai import OpenAI
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
    
    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)
    
    # Load benchmark data
    df = pd.read_csv(args.benchmark_file)
    
    # Load retrieved documents if using RAG or rerank mode
    if args.mode in ["rag", "rerank"]:
        if not args.topk_file:
            raise ValueError("topk file is required for RAG and rerank modes")
        
        with open(args.topk_file, 'r', encoding='utf-8') as f:
            topk_data = json.load(f)
        
        df['retrieved_documents'] = topk_data
        df['topk'] = df['retrieved_documents'].apply(lambda x: x[:4])  # Fixed context length of 4
    
    # Run evaluation
    print(f"Evaluating benchmark dataset in {args.mode} mode...")
    accuracy = evaluate(client, df, "benchmark", args.mode, args.model_name, args.output_file)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"BENCHMARK accuracy: {accuracy:.3f} (mode: {args.mode})")

def set_seed(seed):
    """Set random seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)

def process_answers(answers):
    """Process generated answers to extract letter choices"""
    processed_list = []
    for item in answers:
        # Remove "Answer: " prefix and extract first letter
        answer = item.replace("Answer: ", "").strip()
        if len(answer) > 0:
            processed_list.append(answer[0].upper())
        else:
            processed_list.append('X')  # Mark unprocessable responses as 'X'
    return processed_list

def create_prompts_and_system_message(df, mode):
    """Create prompts and system message based on the inference mode"""
    if mode in ["rag", "rerank"]:
        # RAG/Rerank mode: use context from retrieved documents
        df['context'] = df['topk'].apply(lambda x: '\n\n'.join(x))
        
        system_prompt = """You are an expert Q&A system. Your task is to answer the multiple-choice question below based ONLY on the provided context.
Read the context, question, and options carefully.
Your response MUST be in the format "Answer: [LETTER]", where [LETTER] is the single capital letter of the correct option.
Do not use any prior knowledge. Do not provide explanations or any other text."""
        
        user_prompts = []
        for _, row in df.iterrows():
            user_prompt = f"""### CONTEXT ###
{row['context']}

### QUESTION AND OPTIONS ###
Question: {row['question']}

### FINAL ANSWER ###"""
            user_prompts.append(user_prompt)
    
    else:  # zero-shot mode
        system_prompt = """You are an automated evaluation system. Your sole task is to solve the multiple-choice question below.
Read the question and the provided options carefully.
Your response MUST be in the format "Answer: [LETTER]", where [LETTER] is the single capital letter of the correct option.
Do not write the text of the answer. Do not write any explanations. Do not complete the sentence in the question."""
        
        user_prompts = []
        for _, row in df.iterrows():
            user_prompt = f"""### QUESTION AND OPTIONS ###
Question: {row['question']}

### FINAL ANSWER ###"""
            user_prompts.append(user_prompt)
    
    return system_prompt, user_prompts

def evaluate(client, df, name, mode, model_name, output_file):
    """Evaluate model performance using OpenAI GPT API"""
    print(f"Processing {name} dataset with {len(df)} samples.")
    
    system_prompt, user_prompts = create_prompts_and_system_message(df, mode)
    result = []
    input_token_count, output_token_count = 0, 0
    
    for i, user_prompt in enumerate(tqdm(user_prompts, desc="Generating answers")):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=5,  # Fixed max tokens
                temperature=0,
            )
            
            # Track token usage
            input_token_count += completion.usage.prompt_tokens
            output_token_count += completion.usage.completion_tokens
            
            response = completion.choices[0].message.content.strip()
            result.append(response)
            
            # Calculate running cost (GPT-4o pricing example)
            # running_cost = (5/1000000) * input_token_count + (20/1000000) * output_token_count
            # print(f"Running cost: ${running_cost:.6f}")
            
        except Exception as e:
            print(f"Error processing prompt {i+1}: {e}")
            result.append("ERROR")
    
    # Save results to JSON file
    if output_file:
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    # Calculate final cost
    # final_cost = (5/1000000) * input_token_count + (20/1000000) * output_token_count
    # print(f"Total cost: ${final_cost:.6f}")
    
    # Calculate accuracy
    print(f"Accuracy calculation for {name} dataset")
    gt = df['answer'].tolist()
    generated_answers = process_answers(result)
    
    accuracy = accuracy_score(gt, generated_answers)
    print(f"\n{name.upper()} accuracy: {accuracy:.3f}")
    print("Answer distribution:", Counter(generated_answers))
    
    return round(accuracy, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI GPT inference for music understanding QA")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--benchmark_file", type=str, required=True, help="CSV file containing benchmark questions")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "rag", "rerank"], required=True, help="Inference mode: zero-shot, rag, or rerank")
    parser.add_argument("--topk_file", type=str, help="JSON file containing retrieved documents (required for RAG/rerank)")
    parser.add_argument("--output_file", type=str, help="Output JSON file to save results")
    
    args = parser.parse_args()
    
    main(args)
