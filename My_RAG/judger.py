"""
judger module will enhance the prompt and evaluate the current prompt to see if it's qualified.
"""
from utils import llm_generate


def enhanced_prompt(query, context_chunks, prompt):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt = f"""
    """
    return llm_generate(prompt)


# test the function
if __name__ == "__main__":
    query = ""
    context_chunks = [

    ]
    prompt = ""
    answer = enhanced_prompt(query, context_chunks, prompt)
    print("Generated Answer:", answer)