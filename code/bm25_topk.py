from pyserini.search.lucene import LuceneSearcher
import json
import pandas as pd
from tqdm import tqdm
import time
import argparse

def main(args):
    # Load Lucene index and set BM25 scoring
    print(f"Loading index from: {args.index_path}")
    searcher = LuceneSearcher(args.index_path)
    searcher.set_bm25()
    
    # Load benchmark data
    print(f"Loading benchmark data: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Perform retrieval
    retrieve_documents(searcher, df, args.output_file, args.topk)

def search_topk_texts(searcher, input_text, topk=100):
    """
    Search for top-k most relevant documents using BM25
    
    Args:
        searcher: Pyserini LuceneSearcher object
        input_text: Query text
        topk: Number of top documents to retrieve
        
    Returns:
        List of retrieved document texts
    """
    hits = searcher.search(input_text, k=topk)
    top_texts = []
    for hit in hits:
        doc = searcher.doc(hit.docid)
        raw = doc.get("raw")
        if raw:
            data = json.loads(raw)
            text = data.get("contents", None)
            top_texts.append(text)
        else:
            top_texts.append(None)
    return top_texts

def retrieve_documents(searcher, df, output_file, topk):
    """
    Retrieve top-k documents for all queries in the dataset
    
    Args:
        searcher: Pyserini LuceneSearcher object
        df: DataFrame containing queries
        output_file: Output filename for results
        topk: Number of top documents to retrieve
    """
    result = []
    start_total = time.time()
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving documents"):
        # Extract query from question (remove multiple choice options)
        query = row['question'].split('\nA.')[0]
        top_k_texts = search_topk_texts(searcher, query, topk)
        result.append(top_k_texts)
    
    end_total = time.time()
    total_elapsed = end_total - start_total
    avg_time = total_elapsed / len(df)
    
    print(f"\nTotal time: {total_elapsed:.2f} sec")
    print(f"Average time per query: {avg_time:.4f} sec")
    
    # Save results to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve top-k documents using BM25 with Pyserini")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the Lucene index directory")
    parser.add_argument("--input_file", type=str, required=True, help="CSV file containing benchmark queries")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for retrieval results")
    parser.add_argument("--topk", type=int, default=100, help="Number of top documents to retrieve")
    
    args = parser.parse_args()
    
    main(args)
