from utils import llm_generate


def generate_answer(query, context_chunks, prompt_template):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt_template = prompt_template.replace("{query}", query).replace("{context}", context)
    # prompt = f"""You are an assistant for question-answering tasks. \
    # Use the following pieces of retrieved context to answer the question. \
    # If you don't know the answer, just say that you don't know. \
    # Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
    return llm_generate(prompt_template)


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    prompt = """
    你是一位數據工程師。使用者的需求是從文本中提取特定的結構化數據。

    請閱讀下方的【文本內容】，並根據使用者的指示提取關鍵數據（如：財務指標、年份、具體金額）。
    請務必使用 Markdown 表格 (Table) 或列點的方式呈現，以便閱讀。

    【文本內容】
    {context}

    【提取指令】
    {query}

    【提取結果】
    """
    
    answer = generate_answer(query, context_chunks, prompt)
    print("Generated Answer:", answer)