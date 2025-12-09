from utils import load_jsonl, save_jsonl, expand_query, rerank_chunks
from chunker import chunk_documents 
from hybrid_retriever import create_retriever
# from retriever import create_retriever
# from pyserini_retriever import create_retriever
from generator import generate_answer
from selector import initialize_classifiers, GetPrompt, select_prompt
from judger import enhanced_prompt
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
    chunk_sz = 512 if language == 'en' else 384
    chunks = chunk_documents(docs_for_chunking, language, chunk_size=chunk_sz, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks (Size: {chunk_sz}, Overlap: 100).")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")

    q_classifier, d_classifier = initialize_classifiers()

    for query in tqdm.tqdm(queries, desc="Processing Queries"):
        query_text = query['query']['content']
        
        # 🌟(optional) Query Expansion
        # expanded_query = expand_query(query_text, language)
        
        # 1. Retrieve Candidates (Top-30)
        candidates = retriever.retrieve(query_text, top_k=30)
        
        # 2. Rerank Chunks (Top-5)
        # Verify rerank_chunks is imported
        retrieved_chunks = rerank_chunks(query_text, candidates, language, top_k=5)
        
        """
        classifier
        """
        # domain = d_classifier.predict(query_text)
        # query_type = q_classifier.predict(query_text)
        # gp = GetPrompt(query_text, domain, query_type, retrieved_chunks, language=language)
        # prompt_template = gp.output() 
        prompt_template = select_prompt(query_text, language)

        """
        generator
        """
        answer = generate_answer(query_text, retrieved_chunks, prompt_template, language)

        query["prediction"]["content"] = answer
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
