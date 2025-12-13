import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "result"

def clean_and_parse_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove user-added comments like ",ori_twoG.json -> separate prompt -> retriever" within the incomplete line or trailing commas
    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        # Check if line contains the specific garbage pattern
        # "Words_F1": 0.2848245606667619,ori_twoG.json -> separate prompt -> retriever
        if "ori_" in line and "->" in line:
             line = re.sub(r'(\d+\.?\d*),\s*ori_.*$', r'\1,', line)
        cleaned_lines.append(line)
        
    cleaned_content = "\n".join(cleaned_lines)
    
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {file_path} even after basic cleaning. Error: {e}")
        return None

def extract_metrics(data, score_key):
    if not data or score_key not in data:
        return {}
    return data[score_key]

def plot_comparison(metrics_data, title, output_filename):
    if not metrics_data:
        print(f"No data to plot for {title}")
        return
    
    filenames = list(metrics_data.keys())
    # Assume all files have roughly the same metrics
    first_metrics = metrics_data[filenames[0]]
    metric_names = list(first_metrics.keys())
    
    # Prepare data for plotting
    x = np.arange(len(metric_names))  
    width = 0.8 / len(filenames)  
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, filename in enumerate(filenames):
        values = [metrics_data[filename].get(m, 0) for m in metric_names]
        offset = width * i
        rects = ax.bar(x + offset, values, width, label=filename)

    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(filenames) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.close()

def main():
    if not os.path.exists(RESULT_DIR):
        print(f"Directory {RESULT_DIR} not found.")
        return

    target_files = [f for f in os.listdir(RESULT_DIR) if f.startswith('ori') and f.endswith('.json')]
    target_files.sort()
    
    all_data = {}
    for f in target_files:
        path = os.path.join(RESULT_DIR, f)
        data = clean_and_parse_json(path)
        if data:
            all_data[f] = data
            
    # Prepare data for ZH and EN
    zh_data = {}
    en_data = {}
    
    for fname, data in all_data.items():
        if "score_zh.jsonl" in data:
            zh_data[fname] = data["score_zh.jsonl"]
        if "score_en.jsonl" in data:
            en_data[fname] = data["score_en.jsonl"]
            
    plot_comparison(zh_data, 'Comparison of ZH Metrics', 'comparison_zh.png')
    plot_comparison(en_data, 'Comparison of EN Metrics', 'comparison_en.png')

if __name__ == "__main__":
    main()
