from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents, recursive_chunk_documents
# from bm25 import create_retriever
from hybrid_retriever import create_retriever
from selector import select_prompt
from generator import generate_answer_zh, generate_answer_en, generate_answer
from reranker import create_reranker
import argparse

# english -> zh -> recursive

"""
Top-3: S_Prec=0.0000, S_Rec=0.0000, W_Prec=0.1849, W_Rec=0.6829
Top-5: S_Prec=0.0000, S_Rec=0.0000, W_Prec=0.1642, W_Rec=0.7424
Testing Config: Chunk Size=256, Overlap=100
"""

def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents ori_newG_newChunk_en_1000,100.json ori_reranker.json rerank10-> 8
    print("Chunking documents...")
    if language == "en": 
        # Optimize: Use recursive chunking for EN as well to respect sentence boundaries
        # 2000 400 -> en 
        chunks = recursive_chunk_documents(docs_for_chunking, language, chunk_size=800, chunk_overlap=200) 
    else:# 500 100 good!
        chunks = recursive_chunk_documents(docs_for_chunking, language, chunk_size=128, chunk_overlap=40)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")

    print("Creating reranker...")
    reranker = create_reranker()
    print("Reranker created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']
        # print(f"\nRetrieving chunks for query: '{query_text}'")
        # Optimize: Increase top_k to 10 for English to improve recall of scattered details
        # if language == "zh": 
        #     k = 5 # en zh 5 best 
        # else:
        #     k = 10
        retrieved_chunks = retriever.retrieve(query_text, top_k=30)
        retrieved_chunks = reranker.rerank(query_text, retrieved_chunks, top_k=10)

        # candidates = retriever.retrieve(query_text, top_k=30)
        # print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. Select Prompt
        template_content = select_prompt(query_text)

        # 6. Generate Answer
        # print("Generating answer...") generate_answer_zh is the best till now 
        # 有可能兩個都 recursive 不錯, irrelanvance en 超高
        # template_content = None -> 會 fall back 成原本效果最好的 prompt
        if language == "zh": 
            # zh -> 0.6 0.4 good, en -> 0.5 0.5 good
            # For ZH, use the generator that works best (originally generate_answer)
            answer = generate_answer(query_text, retrieved_chunks[:5], template_content) 
        else: 
            # For EN, use the English-specific generator or default 
            answer = generate_answer_en(query_text, retrieved_chunks[:5], template_content) # retrieval score -> Reranker, generation -> prompt
        
        query["prediction"]["content"] = answer
        # query["prediction"]["references"] = [retrieved_chunks[0]['page_content']] # one chunk, 
        if language == "zh":
            query["prediction"]["references"] = [chunk['page_content'] for chunk in retrieved_chunks[:3]] 
        else:
            query["prediction"]["references"] = [retrieved_chunks[0]['page_content']]
        # 3 or all chunks

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
