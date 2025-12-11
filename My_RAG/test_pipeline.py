import os
import sys

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_jsonl, expand_query, reorder_chunks
from chunker import chunk_documents
from hybrid_retriever import create_retriever
from selector import select_prompt
from generator import generate_answer
from reranker import create_reranker
import tqdm

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def test_pipeline():
    # Redirect stdout to log.txt
    sys.stdout = TeeLogger("log.txt")

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, 'dragonball_dataset', 'dragonball_docs.jsonl')
    queries_path = os.path.join(base_dir, 'dragonball_dataset', 'test_queries_zh.jsonl')

    # 1. Loading Logs
    print(f"Loading docs from {docs_path}...")
    if not os.path.exists(docs_path):
        print(f"Error: {docs_path} not found.")
        return

    docs = load_jsonl(docs_path)
    
    # 2. Chunking
    print("1. Chunking...")
    # Use 'zh' as the target language since the query is Chinese
    language = 'zh'
    # Aligning with main.py settings for Chinese
    chunk_sz = 500 if language == 'en' else 256
    chunk_op = 150 if language == 'en' else 50
    chunks = chunk_documents(docs, language=language, chunk_size=chunk_sz, chunk_overlap=chunk_op)
    print(f"   -> {len(chunks)} chunks created.")

    # 3. Initializing Retriever
    print("2. Initializing Retriever (BM25=0.7, Vector=0.3)...")
    # Weights from user prompt: BM25=0.7, Vector=0.3
    retriever = create_retriever(chunks, language, bm25_weight=0.6, vector_weight=0.4)
    
    # 3. Initializing Reranker
    print("3. Initializing Reranker...")
    reranker = create_reranker()
    
    # Load Queries
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found")
        return
        
    queries = load_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries.")
    
    # Track Metrics
    total_queries = 0
    found_in_retrieval_count = 0
    found_in_rerank_count = 0
    
    limit = 10 # Limit to first 50 queries
    print(f"\n🅰️  Testing First {limit} Queries...")

    for i, query_obj in enumerate(queries[:limit]):
        query_text = query_obj['query']['content']
        # Handle ground truth doc_ids - could be list or single value depending on dataset, usually list
        gt_doc_ids = query_obj.get('ground_truth', {}).get('doc_ids', [])
        gt_answer = query_obj.get('ground_truth', {}).get('content', "N/A")
        query_id = query_obj['query']['query_id']
        
        # DEBUG: Focus on the specific failing query (Query ID 1)
        if query_id == 1: 
            continue

        print(f"\n{'='*50}")
        print(f"[Q{i+1}] ID:{query_id}")
        print(f"Query: {query_text}")
        
        # Step 2: Retrieval (Hybrid)
        candidates = retriever.retrieve(query_text, top_k=100) 
        
        # Step 3: Reranking (Top-5)
        # Rerank top 50 to get top 5
        rerank_input = candidates[:50] 
        reranked_chunks = reranker.rerank(query_text, rerank_input, top_k=10)
        
        print(f"Reordering {len(reranked_chunks)} chunks...")
        reranked_results = reorder_chunks(query_text, reranked_chunks, language)
        print(f" -> {len(reranked_results)} chunks sorted.")
        
        print("--- Top 5 Retrieved & Reranked Chunks ---")
        for idx, chunk in enumerate(reranked_results):
            content = chunk.get('page_content', '').replace('\n', ' ')
            print(f"[{idx+1}] {content[:1000]}") # Print full(er) content
            
        # Step 4: Selector
        prompt_template = select_prompt(query_text, language)
        print(f"Selected Prompt Length: {len(prompt_template)}")
        
        # Step 5: Generator
        print("Generating Answer...")
        try:
            answer = generate_answer(query_text, reranked_results, prompt_template, language) # 
            print(f"🌟 question: {query_text}")
            print(f"\n--- Generated Answer ---")
            print(answer)
            print(f"\n--- Ground Truth ---")
            print(gt_answer)
        except Exception as e:
            print(f"Generation Error: {e}")
            
    # print("\n" + "="*50)
    # print("=== Summary Results ===")
    # print(f"Total Queries Tested: {min(len(queries), limit)}")
    # print(f"Retrieval Recall (Top-100): {found_in_retrieval_count}/{min(len(queries), limit)} ({found_in_retrieval_count/min(len(queries), limit)*100:.1f}%)")
    # print(f"Rerank Recall (Top-5):      {found_in_rerank_count}/{min(len(queries), limit)} ({found_in_rerank_count/min(len(queries), limit)*100:.1f}%)")

if __name__ == "__main__":
    test_pipeline()
