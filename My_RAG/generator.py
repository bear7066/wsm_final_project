from utils import llm_generate

def generate_answer_zh(query, context_chunks):
    # 全部拼接 -> prompt > limit
    # context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])

    MAX_CONTEXT_LEN = 3000
    context_batches = []   
    current_batch = ""     

    for chunk in context_chunks:
        content = chunk['page_content']        
        if len(current_batch) + len(content) < MAX_CONTEXT_LEN:
            if current_batch:
                current_batch += "\n\n"
            current_batch += content
        else:
            if current_batch:
                context_batches.append(current_batch)
            
            # new chunk appear
            current_batch = content

    if current_batch:
        context_batches.append(current_batch)
    
    answers = []
    for i, batch_context in enumerate(context_batches):
        prompt = f"""You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If the answer is not in the context, just say "No relevant information found." \
        Keep the answer concise.\n\nQuestion: {query} \nContext: {batch_context} \nAnswer:\n"""
        
        answer = llm_generate(prompt)
        answers.append(answer)

    return " ".join(answers)


def generate_answer(query, context_chunks):
    # ===========================
    # 1. Smart Batching
    # ===========================
    MAX_CONTEXT_LEN = 3000
    context_batches = []
    current_batch = ""

    for chunk in context_chunks:
        content = chunk['page_content']
        # For English, 3000 chars is roughly 600-800 tokens, which fits easily in most LLM context windows.
        if len(current_batch) + len(content) < MAX_CONTEXT_LEN:
            if current_batch:
                # Use a clear separator in English to prevent context bleeding
                current_batch += "\n\n--- Separator ---\n\n" 
            current_batch += content
        else:
            context_batches.append(current_batch)
            current_batch = content
    
    if current_batch:
        context_batches.append(current_batch)
    
    # ===========================
    # 2. Map Phase: Extraction
    # ===========================
    extracted_infos = []
    
    for batch_context in context_batches:
        # Prompt changed to English
        map_prompt = f"""You are a precise answering assistant. Please read the following context chunks to answer the user's question.
        
        User Question: {query}
        
        Context Chunks:
        {batch_context}
        
        Instructions:
        1. If the context contains the answer or relevant clues, extract the key information and answer concisely.
        2. If the context is **completely irrelevant** to the question, you must strictly output "NO_INFO" only. Do not provide any other text.
        3. Answer only based on the provided context. Do not fabricate information.
        
        Answer:"""
        
        # Assume llm_generate is your function to call the LLM
        # Suggestion: Set temperature to 0.1 for the Map phase for consistency
        partial_answer = llm_generate(map_prompt).strip()
        
        # Filter logic remains the same
        if "NO_INFO" not in partial_answer and len(partial_answer) > 0:
            extracted_infos.append(partial_answer)

    # ===========================
    # 3. Reduce Phase: Synthesis
    # ===========================
    
    # Case A: No information found in any batch
    if not extracted_infos:
        return "Sorry, no relevant information found in the provided documents."
    
    # Case B: Only one source found (Return directly to save cost/latency)
    if len(extracted_infos) == 1:
        return extracted_infos[0]
    
    # Case C: Multiple sources found (Synthesis required)
    combined_info = "\n".join([f"Source {i+1}: {info}" for i, info in enumerate(extracted_infos)])
    
    reduce_prompt = f"""You are a professional integration assistant. The user asked a question, and we have retrieved the following partial answers from different context chunks.
    Please synthesize this information into a coherent, logically connected final answer.
    
    User Question: {query}
    
    Collected Partial Answers:
    {combined_info}
    
    Integration Requirements:
    1. Remove duplicate information.
    2. Connect scattered information points into a fluent paragraph.
    3. If there are contradictions between the chunks, point them out in the answer.
    4. Provide the final answer directly without opening greetings.
    
    Final Synthesized Answer:"""
    
    final_answer = llm_generate(reduce_prompt)
    
    return final_answer


def generate_answer_en(query, context_chunks):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt = f"""You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
    try:
        response = llm_generate(prompt)
        return response
    except Exception as e:
        return f"Error using Ollama Python client: {e}"


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)