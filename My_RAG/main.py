from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents, recursive_chunk_documents
# from bm25 import create_retriever
from hybrid_retriever import create_retriever
from selector import select_prompt
from generator import generate_answer_zh, generate_answer_en, generate_answer
import argparse

# english -> zh -> recursive

def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    if language == "en": 
        chunks = chunk_documents(docs_for_chunking, language, chunk_size=2000, chunk_overlap=400)
    else:
        chunks = recursive_chunk_documents(docs_for_chunking, language, chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']
        # print(f"\nRetrieving chunks for query: '{query_text}'")
        retrieved_chunks = retriever.retrieve(query_text)
        # print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. Select Prompt
        # template_content = select_prompt(query_text)

        # 6. Generate Answer
        # print("Generating answer...") generate_answer_zh is the best till now 
        # 有可能兩個都 recursive 不錯, irrelanvance en 超高
        """
        1. recursive zh, normal en. 2. 斷詞->bm25, lowercase ->gemma 
        """
        if language == "zh":
            answer = generate_answer(query_text, retrieved_chunks) # zh -> this generator is good 
        else:
            answer = generate_answer_zh(query_text, retrieved_chunks)
        
        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [retrieved_chunks[0]['page_content']] # exp -> all, exp -> 2, 0 -> best,  [chunk['page_content'] for chunk in retrieved_chunks[:2]] 

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
