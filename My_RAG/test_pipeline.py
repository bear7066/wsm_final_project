import sys
import os

# Ensure My_RAG is in the path
sys.path.append(os.path.join(os.getcwd(), 'My_RAG'))

from utils import load_jsonl
from chunker import recursive_chunk_documents
from hybrid_retriever import create_retriever
from generator import generate_answer_en

def main():
    docs_path = "dragonball_dataset/dragonball_docs.jsonl"
    query_path = "dragonball_dataset/test_queries_en.jsonl"
    
    print(f"Loading documents from {docs_path}...")
    docs = load_jsonl(docs_path)

    print(f"Loading queries from {query_path}...")
    queries = load_jsonl(query_path)
    # Take first 5 queries for debugging
    test_queries = queries[:5]

    # strict configuration from main.py for EN
    # main.py line 29: recursive_chunk_documents, chunk_size=2000, chunk_overlap=400
    chunk_configs = [(2000, 400)]

    for chunk_size, chunk_overlap in chunk_configs:
        print("\n" + "#"*50)
        print(f"TESTING CONFIG (Optimized): Chunk Size={chunk_size}, Overlap={chunk_overlap}")
        print("#"*50)
        
        print(f"Chunking documents (en) using recursive_chunk_documents...")
        chunks = recursive_chunk_documents(docs, language="en", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Created {len(chunks)} chunks.")
        
        print("Creating retriever...")
        retriever = create_retriever(chunks, language="en")
        
        print("\n" + "="*50)
        print("STARTING TEST PIPELINE")
        print("="*50)
        
        for i, q in enumerate(test_queries):
            query_text = q['query']['content']
            
            print(f"\nQuery {i+1}: {query_text}")
            
            # Retrieve
            # main.py now uses top_k=10 for EN
            retrieved_chunks = retriever.retrieve(query_text, top_k=10)
            
            print(f"Top 5 Retrieved Chunks (out of 10):")
            for j, chunk in enumerate(retrieved_chunks[:5]):
                content = chunk.get('page_content', '')
                # Display first 200 chars
                display_content = content[:200] + "..." if len(content) > 200 else content
                print(f"  [{j+1}]: {display_content}")
                print(f"       (Source ID: {chunk.get('metadata', {}).get('doc_id', 'N/A')})")
                
            # Generate
            # main.py uses generate_answer_en
            print("Generating answer (using generate_answer_en)...")
            answer = generate_answer_en(query_text, retrieved_chunks)
            print(f"🌟Generated Answer: {answer}\n")
            
            # Ground Truth Display
            gt_content = "N/A"
            if 'ground_truth' in q and isinstance(q['ground_truth'], dict):
                 gt_content = q['ground_truth'].get('content', 'N/A')
            elif 'ground_truth' in q:
                 gt_content = q['ground_truth']
            
            print(f"🌟 Ground Truth: {gt_content}")
            print("-" * 50)

class Logger(object):
    def __init__(self, filename="log_test.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    # sys.stdout = Logger() # Optional: Disable logger redirection to see output in terminal directly for now
    main()
