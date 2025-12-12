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


def generate_answer(query, context_chunks, template_content=None):
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
    # 2. Generation Phase (Map/Reduce or Direct)
    # ===========================
    
    # If a template is provided, we can try to use it directly on the chunks.
    # However, if there are multiple batches, we still need a strategy.
    # Strategy: 
    # - If 1 batch: Use template directly.
    # - If > 1 batch: Use template on each batch (Map) then Synthesize? 
    #   OR Use default Map then Template on Reduce?
    #   Let's stick to the Plan: Use Template for Final Synthesis if > 1 batch, or Direct if 1 batch.
    
    # But wait, "data_extraction" templates might be better suited for the Map phase.
    # Evaluating complexity... The user wants "appropriate prompt template".
    # Let's apply the template to the FINAL generation step. 
    # If there are multiple batches, we first extract "relevant info" (Map) using a generic prompt,
    # then use the Custom Template for the final answer (Reduce).
    
    extracted_infos = []
    
    # If only 1 batch, we skip the Map phase and go straight to generation with the template
    if len(context_batches) == 1:
        # Direct Generation
        context = context_batches[0]
        if template_content:
            # Inject context and query into template
            # Templates have {context} and {query} placeholders
            final_prompt = template_content.format(context=context, query=query)
        else:
            # Default Prompt
             final_prompt = f"""You are a precise answering assistant. Please read the following context chunks to answer the user's question.
            
            User Question: {query}
            
            Context Chunks:
            {context}
            
            Instructions:
            1. If the context contains the answer or relevant clues, extract the key information and answer concisely.
            2. If the context is **completely irrelevant** to the question, you must strictly output "NO_INFO" only. Do not provide any other text.
            3. Answer only based on the provided context. Do not fabricate information.
            
            Answer:"""
            
        return llm_generate(final_prompt).strip()

    # If > 1 batch, we continue with Map-Reduce logic
    
    # --- Map Phase (Extraction) ---
    for batch_context in context_batches:
        # Use a generic extraction prompt here to gather info
        # (Unless we want to apply the custom template here too? 
        #  If the custom template is "Data Extraction", maybe we should?
        #  But "Summary" might not work well on partial chunks.
        #  Let's keep Map generic for safety, and apply Custom Template on Reduce.)
        
        map_prompt = f"""You are a precise answering assistant. Please read the following context chunks to answer the user's question.
        
        User Question: {query}
        
        Context Chunks:
        {batch_context}
        
        Instructions:
        1. If the context contains the answer or relevant clues, extract the key information and answer concisely.
        2. If the context is **completely irrelevant** to the question, you must strictly output "NO_INFO" only. Do not provide any other text.
        3. Answer only based on the provided context. Do not fabricate information.
        
        Answer:"""
        
        partial_answer = llm_generate(map_prompt).strip()
        
        if "NO_INFO" not in partial_answer and len(partial_answer) > 0:
            extracted_infos.append(partial_answer)

    # --- Reduce Phase (Synthesis) ---
    
    # Case A: No information found in any batch
    if not extracted_infos:
        return "Sorry, no relevant information found in the provided documents."
    
    # Case B: Only one source found (Return directly to save cost/latency)
    if len(extracted_infos) == 1:
        # If we have a custom template, we might want to re-format this single piece of info?
        # But extracted info is already an answer. 
        # Let's just return it for efficiency, unless the user strictly enforces template format.
        # Given "data_extraction" often wants structured output, maybe we should re-run?
        # For now, return as is.
        return extracted_infos[0]
    
    # Case C: Multiple sources found (Synthesis required)
    combined_info = "\n".join([f"Source {i+1}: {info}" for i, info in enumerate(extracted_infos)])
    
    if template_content:
        # Use Custom Template for Synthesis
        # We treat "combined_info" as the context
        final_prompt = template_content.format(context=combined_info, query=query)
    else:
        # Default Reduce Prompt
        final_prompt = f"""You are a professional integration assistant. The user asked a question, and we have retrieved the following partial answers from different context chunks.
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
    
    final_answer = llm_generate(final_prompt)
    
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