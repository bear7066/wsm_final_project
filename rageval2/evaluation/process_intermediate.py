import json
import os
from typing import Dict, List, Any
import jieba
import pysbd
import numpy as np

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def get_sentences(text, language='en'):
    if not text:
        return []
    seg = pysbd.Segmenter(language=language, clean=False)
    return seg.segment(text)

def get_words(text, language='en'):
    if not text:
        return []
    if language == 'zh':
        return list(jieba.cut(text))
    else:
        # Simple whitespace splitting for English
        return text.split()

def calculate_overlap_metrics(prediction, reference, unit_type='words', language='en'):
    if unit_type == 'sentences':
        pred_units = set(get_sentences(prediction, language))
        ref_units = set(get_sentences(reference, language))
    else:
        pred_units = set(get_words(prediction, language))
        ref_units = set(get_words(reference, language))
    
    if not pred_units and not ref_units:
        return 1.0, 1.0, 1.0 # Perfect match if both empty
    if not pred_units:
        return 0.0, 0.0, 0.0 
    if not ref_units:
        return 0.0, 0.0, 0.0 

    intersection = pred_units.intersection(ref_units)
    
    precision = len(intersection) / len(pred_units) if pred_units else 0.0
    recall = len(intersection) / len(ref_units) if ref_units else 0.0
    f1 = calculate_f1(precision, recall)
    
    return precision, recall, f1

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path} at line {i+1}: {e}")
    return data

def calculate_averages(data: List[Dict[str, Any]], metric_list: List[str]) -> Dict[str, float]:
    metric_sums = {metric: 0 for metric in metric_list}
    metric_counts = {metric: 0 for metric in metric_list}
    
    for item in data:
        for metric in metric_list:
            if metric in item:
                metric_sums[metric] += item[metric]
                metric_counts[metric] += 1
    #only return the average of the metrics that the metric_counts is not 0
    return {metric: metric_sums[metric] / metric_counts[metric] for metric in metric_list if metric_counts[metric] != 0}


def process_folder(folder_path: str, output_file: str):
    results = {}
    
    # Extended metric list including new ones
    base_metrics = ['EIR', 'Precision', 'Recall', 'ROUGELScore', "completeness", "hallucination", "irrelevance"]
    new_metrics = [
        "Sentences_Precision", "Sentences_Recall", "Sentences_F1",
        "Words_Precision", "Words_Recall", "Words_F1"
    ]
    all_metrics_to_avg = base_metrics + new_metrics

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            data = load_jsonl(file_path)
            
            # Determine language from content or filename if possible, defaulting to 'zh' if unsure or based on filename
            language = 'zh' if 'zh' in filename else 'en'

            # Calculate new metrics for each item
            for item in data:
                prediction = item.get("prediction", {}).get("content", "")
                ground_truth = item.get("ground_truth", {}).get("content", "")
                
                # Sentence Metrics
                s_prec, s_rec, s_f1 = calculate_overlap_metrics(prediction, ground_truth, 'sentences', language)
                item["Sentences_Precision"] = s_prec
                item["Sentences_Recall"] = s_rec
                item["Sentences_F1"] = s_f1
                
                # Word Metrics
                w_prec, w_rec, w_f1 = calculate_overlap_metrics(prediction, ground_truth, 'words', language)
                item["Words_Precision"] = w_prec
                item["Words_Recall"] = w_rec
                item["Words_F1"] = w_f1

            averages = calculate_averages(data, all_metrics_to_avg)
            
            # Calculate derived metrics
            # Factual_Score = Completeness - Hallucination
            completeness = averages.get("completeness", 0.0)
            hallucination = averages.get("hallucination", 0.0)
            averages["Factual_Score"] = completeness - hallucination
            
            # Generation_Total_Score = Mean(ROUGELScore, Factual_Score, Words_F1)
            rouge = averages.get("ROUGELScore", 0.0)
            factual = averages.get("Factual_Score", 0.0)
            words_f1 = averages.get("Words_F1", 0.0)
            averages["Generation_Total_Score"] = np.mean([rouge, factual, words_f1])
            
            # Retrieval_Total_Score = Mean(Sentences_F1, Words_F1) -- per previous decision
            sentences_f1 = averages.get("Sentences_F1", 0.0)
            averages["Retrieval_Total_Score"] = (sentences_f1 + words_f1) / 2

            results[filename] = averages
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    folder_path = './result'
    output_file = './result/final_result.json'
    
    process_folder(folder_path, output_file)
    print(f"Results saved to {output_file}")