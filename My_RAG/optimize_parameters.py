
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import pysbd
import jieba

# Import project modules (assuming script is run from project root)
from utils import load_jsonl
from chunker import recursive_chunk_documents
from hybrid_retriever import create_retriever

# Reuse metric calculation logic
def get_sentences(text, language='en'):
    if not text:
        return []
    if language == 'zh':
        # Simple split for Chinese sentences if pysbd usually focuses on EN
        # But pysbd claims multi-language. Let's stick to simple split for consistency if pysbd fails for ZH
        # Or better, replicate process_intermediate.py logic
        seg = pysbd.Segmenter(language=language, clean=False)
        return seg.segment(text)
    else:
        seg = pysbd.Segmenter(language=language, clean=False)
        return seg.segment(text)

def get_words(text, language='en'):
    if not text:
        return []
    if language == 'zh':
        return list(jieba.cut(text))
    else:
        return text.split()

def calculate_metrics(prediction, reference, language='en'):
    # Sentence Metrics
    pred_sents = set(get_sentences(prediction, language))
    ref_sents = set(get_sentences(reference, language))
    
    if not pred_sents and not ref_sents:
        s_prec, s_rec = 1.0, 1.0
    elif not pred_sents:
        s_prec, s_rec = 0.0, 0.0
    elif not ref_sents:
        s_prec, s_rec = 0.0, 0.0
    else:
        intersection = pred_sents.intersection(ref_sents)
        s_prec = len(intersection) / len(pred_sents)
        s_rec = len(intersection) / len(ref_sents)

    # Word Metrics
    pred_words = set(get_words(prediction, language))
    ref_words = set(get_words(reference, language))
    
    if not pred_words and not ref_words:
        w_prec, w_rec = 1.0, 1.0
    elif not pred_words:
        w_prec, w_rec = 0.0, 0.0
    elif not ref_words:
        w_prec, w_rec = 0.0, 0.0
    else:
        intersection = pred_words.intersection(ref_words)
        w_prec = len(intersection) / len(pred_words)
        w_rec = len(intersection) / len(ref_words)
        
    return s_prec, s_rec, w_prec, w_rec

def run_experiment(docs_path, query_path, language, chunk_sizes, overlaps, top_ks):
    print(f"\n--- Starting Experiment for Language: {language} ---")
    
    docs = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    
    results = []

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                continue
                
            print(f"\nTesting Config: Chunk Size={chunk_size}, Overlap={overlap}")
            
            # 1. Chunking
            chunks = recursive_chunk_documents(docs, language=language, chunk_size=chunk_size, chunk_overlap=overlap)
            print(f"  Generated {len(chunks)} chunks.")
            
            # 2. Build Retriever
            # Note: This is expensive as it computes embeddings every time.
            retriever = create_retriever(chunks, language)
            
            # 3. Retrieve & Evaluate for each Top K
            # We compute all retrievals once (max k) then slice for efficiency? 
            # But the retrieve method returns top_k. Let's just use max(top_ks)
            
            max_k = max(top_ks)
            
            # Store accumulators for each k
            # Key: k, Value: {s_prec_sum, s_rec_sum, w_prec_sum, w_rec_sum}
            metrics_sum = {k: {'s_prec': 0.0, 's_rec': 0.0, 'w_prec': 0.0, 'w_rec': 0.0} for k in top_ks}
            
            for query_item in tqdm(queries, desc="Evaluating queries"):
                query_text = query_item['query']['content']
                ground_truth = query_item['ground_truth']['content'] # Assuming simple string ground truth
                # If ground_truth is complex, adjust. Usually 'content' is the answer text.
                # However, for RETRIEVAL metrics, we usually compare against reference DOCUMENTS or Keypoints?
                # The user asked for "Sentences precision... word precision" which implies comparing the RETRIEVED TEXT against GROUND TRUTH CONTENT.
                # Just like 'process_intermediate.py' does: prediction vs ground_truth.
                
                retrieved_chunks = retriever.retrieve(query_text, top_k=max_k)
                
                for k in top_ks:
                    # Take top k chunks
                    current_chunks = retrieved_chunks[:k]
                    # Concat content
                    prediction_text = " ".join([c['page_content'] for c in current_chunks])
                    
                    s_p, s_r, w_p, w_r = calculate_metrics(prediction_text, ground_truth, language)
                    
                    metrics_sum[k]['s_prec'] += s_p
                    metrics_sum[k]['s_rec'] += s_r
                    metrics_sum[k]['w_prec'] += w_p
                    metrics_sum[k]['w_rec'] += w_r
            
            # 4. Average and Store
            num_queries = len(queries)
            for k in top_ks:
                res = {
                    'language': language,
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'top_k': k,
                    'sentences_precision': metrics_sum[k]['s_prec'] / num_queries,
                    'sentences_recall': metrics_sum[k]['s_rec'] / num_queries,
                    'words_precision': metrics_sum[k]['w_prec'] / num_queries,
                    'words_recall': metrics_sum[k]['w_rec'] / num_queries
                }
                results.append(res)
                print(f"  Top-{k}: S_Prec={res['sentences_precision']:.4f}, S_Rec={res['sentences_recall']:.4f}, W_Prec={res['words_precision']:.4f}, W_Rec={res['words_recall']:.4f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_path', default='./dragonball_dataset/dragonball_docs.jsonl')
    parser.add_argument('--query_path_en', default='./dragonball_dataset/test_queries_en.jsonl')
    parser.add_argument('--query_path_zh', default='./dragonball_dataset/test_queries_zh.jsonl')
    args = parser.parse_args()

    # Define Experiment Grid
    # You can modify this
    CHUNK_SIZES = [256, 512, 1024]
    OVERLAPS = [100]
    TOP_KS = [3, 5]
    
    all_results = []
    
    # Run for ZH
    all_results.extend(run_experiment(
        args.docs_path, 
        args.query_path_zh, 
        "zh", 
        CHUNK_SIZES, 
        OVERLAPS, 
        TOP_KS
    ))

    # Run for EN
    # Uncomment if needed, or run both
    all_results.extend(run_experiment(
        args.docs_path, 
        args.query_path_en, 
        "en", 
        CHUNK_SIZES, 
        OVERLAPS, 
        TOP_KS
    ))
    
    # Print Summary Table
    print("\n\n=== Final Experiment Results ===")
    print(f"{'Lang':<5} {'Chunk':<6} {'Overlap':<8} {'TopK':<5} {'S_Prec':<8} {'S_Rec':<8} {'W_Prec':<8} {'W_Rec':<8}")
    for r in all_results:
        print(f"{r['language']:<5} {r['chunk_size']:<6} {r['overlap']:<8} {r['top_k']:<5} {r['sentences_precision']:.4f}   {r['sentences_recall']:.4f}   {r['words_precision']:.4f}   {r['words_recall']:.4f}")

    # Save to file
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
