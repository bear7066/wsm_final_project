from utils import load_jsonl, save_jsonl, expand_query, reorder_chunks
from chunker import chunk_documents, chunk_documents_original 
from hybrid_retriever import create_retriever
# from retriever import create_retriever
# from pyserini_retriever import create_retriever
from reranker import create_reranker
from generator import generate_answer
from selector import  select_prompt
import argparse, tqdm


def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents: pure chunk
    print("Chunking documents...")
    # Based on analysis: English P99 ref length is ~325 chars, Max is 453. 
    # 512 ensures full context coverage. Chinese can be smaller (384).
    chunk_sz = 500 if language == 'en' else 256
    chunk_op = 150 if language == 'en' else 50
    if language == 'en':
        print(f"Using chunk_documents_original for {language}")
        chunks = chunk_documents(docs_for_chunking, language, chunk_size=chunk_sz, chunk_overlap=chunk_op)
    else:
        print(f"Using chunk_documents for {language}")
        chunks = chunk_documents(docs_for_chunking, language, chunk_size=chunk_sz, chunk_overlap=chunk_op)
    print(f"Created {len(chunks)} chunks (Size: {chunk_sz}, Overlap: {chunk_op}).")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")

    # 4. Create Reranker
    print("Creating reranker...")
    reranker = create_reranker()
    print("Reranker created successfully.")

    for query in tqdm.tqdm(queries, desc="Processing Queries"):
        query_text = query['query']['content']        
        # 🌟(optional) Query Expansion
        # expanded_query = expand_query(query_text, language)
        
        # 1. Retrieve Candidates (Top-100 for high recall)
        candidates = retriever.retrieve(query_text, top_k=50)
        # top_candidates = candidates[:5] 
        if language == 'zh':
            # 2. Rerank Candidates (Top-5 for high precision)
            top_candidates = reranker.rerank(query_text, candidates, top_k=10)
        else:
            top_candidates = candidates[:5]
        
        # 3. Reorder Candidates (LLM Sort) -> put most relevant chunks first
        top_candidates = reorder_chunks(query_text, top_candidates, language)

        """
        classifier
        """
        # domain = d_classifier.predict(query_text)
        # query_type = q_classifier.predict(query_text)
        # gp = GetPrompt(query_text, domain, query_type, retrieved_chunks, language=language)
        # prompt_template = gp.output() 
        # 1. Select Prompt
        prompt_template = select_prompt(query_text, language)

        """
        generator
        """
        # 3. Use top 5 chunks to generate answer
        answer = generate_answer(query_text, top_candidates, prompt_template, language)
        # answer = generate_answer(query_text, top_candidates, prompt_template, language)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [chunk['page_content'] for chunk in top_candidates]

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
