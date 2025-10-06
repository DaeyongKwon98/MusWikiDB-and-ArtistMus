from google import genai
from google.genai import types
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
    
    # Initialize Gemini client
    client = genai.Client(api_key=args.api_key)
    
    # Load benchmark data
    df = pd.read_csv(args.benchmark_file)
    
    # Load retrieved documents if using RAG or rerank mode
    if args.mode in ["rag", "rerank"]:
        if not args.topk_file:
            raise ValueError("topk file is required for RAG and rerank modes")
        
        with open(args.topk_file, 'r', encoding='utf-8') as f:
            topk_data = json.load(f)
        
        df['retrieved_docs'] = topk_data
        df['topk'] = df['retrieved_docs'].apply(lambda x: x[:4] if isinstance(x, list) else [])
    
    # Run evaluation
    print(f"Evaluating benchmark dataset in {args.mode} mode...")
    accuracy = evaluate(client, df, "benchmark", args.mode, args.model_name, args.max_new_tokens, args.output_file)
    
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
Please answer as format below:
Answer: answer"""
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
Answer: answer"""
            prompts.append(prompt)
    
    return prompts

def evaluate(client, df, name, mode, model_name, max_new_tokens, output_file):
    """Evaluate model performance using Gemini API"""
    print(f"Processing {name} dataset with {len(df)} samples.")
    
    prompts = create_prompts(df, mode)
    result = []
    input_token_count, output_token_count = 0, 0
    
    for i, prompt_text in enumerate(tqdm(prompts, desc="Generating answers")):
        prompt = types.Part.from_text(text=prompt_text)
        user_content = types.Content(role="user", parts=[prompt])
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[user_content],
                config=types.GenerateContentConfig(
                    temperature=0,
                    maxOutputTokens=max_new_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking
                ),
            )
            
            # Track token usage
            prompt_tokens = response.usage_metadata.prompt_token_count
            total_tokens = response.usage_metadata.total_token_count
            output_tokens = total_tokens - prompt_tokens
            
            input_token_count += prompt_tokens
            output_token_count += output_tokens
            
            result.append(response.text)
            print(f"Response {i+1}: {response.text}")
            
        except Exception as e:
            print(f"Error processing prompt {i+1}: {e}")
            result.append("ERROR")
    
    # Save results to JSON file
    if output_file:
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    # Calculate cost (example pricing for Gemini)
    # price = (0.3/1000000) * input_token_count + (2.5/1000000) * output_token_count
    # print(f"Estimated cost: ${price:.5f}")
    
    # Calculate accuracy
    print(f"Accuracy calculation for {name} dataset")
    gt = df['answer'].tolist()
    generated_answers = process_answers(result)
    
    accuracy = accuracy_score(gt, generated_answers)
    print(f"\n{name.upper()} accuracy: {accuracy:.3f}")
    print("Answer distribution:", Counter(generated_answers))
    
    return round(accuracy, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini inference for music understanding QA")
    parser.add_argument("--api_key", type=str, required=True, help="Google Gemini API key")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--benchmark_file", type=str, required=True, help="CSV file containing benchmark questions")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "rag", "rerank"], required=True, help="Inference mode: zero-shot, rag, or rerank")
    parser.add_argument("--topk_file", type=str, help="JSON file containing retrieved documents (required for RAG/rerank)")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum number of new tokens to generate")
    parser.add_argument("--output_file", type=str, help="Output JSON file to save results")
    
    args = parser.parse_args()
    
    main(args)