import pandas as pd
import json
from tqdm import tqdm
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main(args):
    # GPU configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer with FP16 precision
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device).half()
    model.eval()
    
    # Load benchmark data
    print(f"Loading benchmark data: {args.benchmark_file}")
    df = pd.read_csv(args.benchmark_file)
    
    # Extract query from question (remove multiple choice options)
    df['query'] = df['question'].apply(lambda x: x.split('\nA.')[0])
    
    # Load retrieval results
    print(f"Loading retrieval results: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Perform reranking
    rerank(data, df, args.output_file, args.batch_size, args.max_length, model, tokenizer, device)

def rerank(data, df, filename, batch_size, max_length, model, tokenizer, device):
    """
    Rerank retrieved chunks using BGE reranker model
    
    Args:
        data: List of top-k chunks for each query
        df: DataFrame containing queries
        filename: Output filename for reranked results
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization
        model: Loaded reranker model
        tokenizer: Loaded tokenizer
        device: Computing device (CPU/GPU)
    """
    result = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Reranking batches"):
        batch = df.iloc[i:i + batch_size]
        all_pairs = []
        batch_chunks = []

        # Create query-chunk pairs for batch processing
        for j, row in batch.iterrows():
            top_chunks = data[j]
            pairs = [[row['query'], chunk] for chunk in top_chunks]
            all_pairs.extend(pairs)
            batch_chunks.append(top_chunks)

        # Compute relevance scores using GPU with FP16
        with torch.no_grad():
            inputs = tokenizer(all_pairs, padding=True, truncation=True, 
                             return_tensors='pt', max_length=max_length).to(device)
            scores = model(**inputs, return_dict=True).logits.view(-1).float()

        # Reorder chunks based on relevance scores
        scores = scores.view(len(batch), -1)  # Reshape scores by batch
        sorted_indices = torch.argsort(scores, descending=True, dim=1)

        # Apply reranking to each query in batch
        for idx, top_chunks in enumerate(batch_chunks):
            sorted_chunks = [top_chunks[i] for i in sorted_indices[idx]]
            result.append(sorted_chunks)

    # Save reranked results to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Reranked results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank retrieved chunks using BGE reranker model")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file containing top-k chunks")
    parser.add_argument("--benchmark_file", type=str, required=True, help="CSV file containing benchmark queries")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for reranked results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-large", help="Reranker model name")
    
    args = parser.parse_args()
    
    main(args)
