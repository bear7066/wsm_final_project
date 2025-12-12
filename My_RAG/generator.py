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