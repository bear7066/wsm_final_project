import sys
import os

# Ensure My_RAG is in the path
sys.path.append(os.path.join(os.getcwd(), 'My_RAG'))

from utils import load_jsonl
from chunker import chunk_documents
from hybrid_retriever import create_retriever
from generator import generate_answer

def main():
    docs_path = "dragonball_dataset/dragonball_docs.jsonl"
    query_path = "dragonball_dataset/test_queries_en.jsonl"
    
    print(f"Loading documents from {docs_path}...")
    docs = load_jsonl(docs_path)

    print(f"Loading queries from {query_path}...")
    queries = load_jsonl(query_path)
    # Take first 5 queries
    test_queries = queries[:5]

    chunk_configs = [(500, 50), (1000, 200), (1500, 300), (2000, 400)]

    for chunk_size, chunk_overlap in chunk_configs:
        print("\n" + "#"*50)
        print(f"TESTING CONFIG: Chunk Size={chunk_size}, Overlap={chunk_overlap}")
        print("#"*50)
        
        print(f"Chunking documents (en)...")
        chunks = chunk_documents(docs, language="en", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
            retrieved_chunks = retriever.retrieve(query_text, top_k=5)
            print(f"Top 3 Retrieved Chunks:")
            for j, chunk in enumerate(retrieved_chunks):
                content = chunk.get('page_content', '')
                # Truncate for display
                display_content = content[:150] + "..." if len(content) > 150 else content
                print(f"  [{j+1}] (Score: ?): {display_content}")
                
            # Generate
            answer = generate_answer(query_text, retrieved_chunks)
            print(f"🌟Generated Answer: {answer}\n")
            
            if 'ground_truth' in q and 'content' in q['ground_truth']:
                 print(f"🌟 Ground Truth: {q['ground_truth']['content']}")
            elif 'response' in q:
                 print(f"🌟 Ground Truth: {q['response']}")
            elif 'answer' in q:
                 print(f"🌟 Ground Truth: {q['answer']['content'] if isinstance(q['answer'], dict) and 'content' in q['answer'] else q['answer']}")
            else:
                 print("Ground Truth: N/A")
                 
            print("-" * 50)

class Logger(object):
    def __init__(self, filename="log.txt"):
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
    sys.stdout = Logger()
    main()
