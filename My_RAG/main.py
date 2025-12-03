from utils import load_jsonl, save_jsonl, expand_query, rerank_chunks
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from selector import select_prompt
from judger import enhanced_prompt
import argparse, tqdm
# [GIN] 2025/12/03 - 23:28:56 | 200 | 20.567621792s |       127.0.0.1 | POST     "/api/generate"
#time=2025-12-03T23:28:56.441+08:00 level=WARN source=runner.go:128 msg="truncating input prompt" limit=4096 prompt=4479 keep=4 new=4096

# 500 150 -> top5

def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")


    for query in tqdm.tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']
        
        # 🌟(optional) Query Expansion
        # expanded_query = expand_query(query_text, language)
        # full_query = f"{query_text} {expanded_query}"
        
        # print(f"\nRetrieving chunks for query: '{query_text}'")
        # Retrieve Top-10 candidates first
        candidate_chunks = retriever.retrieve(query_text, top_k=10)
        
        # Rerank to get Top-5
        retrieved_chunks = rerank_chunks(query_text, candidate_chunks, language, top_k=5)
        # print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. 
        # Select prompt template 
        # (optional) enhance prompt        
        # Generate Answer
        prompt_template = select_prompt(query_text, retrieved_chunks) 
        # final_prompt = enhanced_prompt(query_text, retrieved_chunks, prompt_template)
        answer = generate_answer(query_text, retrieved_chunks, prompt_template, language)

        query["prediction"]["content"] = answer
        # Modified: Save all retrieved chunks to improve Recall (previously only saved top-1)
        # query["prediction"]["references"] = [retrieved_chunks[0]['page_content']]
        query["prediction"]["references"] = [chunk['page_content'] for chunk in retrieved_chunks]

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
