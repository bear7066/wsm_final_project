import json
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunker import chunk_documents
from hybrid_retriever import create_retriever

def reproduce():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, 'dragonball_dataset', 'dragonball_docs.jsonl')
    target_doc = None
    
    print(f"Searching for document in {docs_path}...")
    with open(docs_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            # The specific text from the grep
            if "The first sub-event was the appointment of a new CEO in January 2021" in doc.get('content', ''):
                target_doc = doc
                break
    
    if not target_doc:
        print("Could not find target document!")
        return

    print(f"Found document ID: {target_doc.get('id', 'unknown')}")

    language = 'en'
    configs = [(500, 150), (600, 200), (800, 200), (1000, 200)]
    
    for sz, op in configs:
        print(f"\nTesting Chunk Size={sz}, Overlap={op}...")
        chunks = chunk_documents([target_doc], language=language, chunk_size=sz, chunk_overlap=op)
        
        found_good_chunk = False
        for i, chunk in enumerate(chunks):
            content = chunk['page_content']
            if "The first sub-event was the appointment of a new CEO in January 2021" in content:
                has_entity = "Green Fields Agriculture Ltd." in content
                print(f"  Chunk {i} has answer. Has Entity? {has_entity}")
                if has_entity:
                    print("  ✅ SUCCESS: Answer and Entity combined!")
                    found_good_chunk = True
                    
        if not found_good_chunk:
             print("  ❌ No single chunk contains both Answer and Entity.")

if __name__ == "__main__":
    reproduce()
