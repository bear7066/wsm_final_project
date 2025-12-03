from ollama import Client
from pathlib import Path
import jsonlines
import yaml


def load_jsonl(file_path):
    docs = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            docs.append(obj)
    return docs


def save_jsonl(file_path, data):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)


def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]
    
def llm_generate(prompt: str, model: str = "granite4:3b") -> str:
    """
    Sends a prompt to the Ollama model and returns the response.

    Args:
        prompt: The prompt to send to the model.
        model: The name of the model to use.

    Returns:
        The model's response as a string.
    """
    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        """ 
            num_ctx, temperature, num_predict
            explaination in Final_Tutorials 4
        """
        response = client.generate(
            model=model, 
            prompt=prompt, 
            stream=False, 
            options={
                "temperature": 0.0
            }
        )
        return response.get("response", "No response from model.")
    except Exception as e:
        return f"Error using Ollama Python client: {e}"
    

if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    
    context_text = "\n".join([chunk["page_content"] for chunk in context_chunks])
    full_prompt = f"""
    Based on the following context, answer the question.
    
    Context:
    {context_text}
    
    Question: 
    {query}
    """

    print("Sending prompt to LLM...")
    answer = llm_generate(full_prompt)
    print("Generated Answer:", answer)


def expand_query(query_text, language):
    if language == 'zh':
        prompt = f"You are a search query optimizer. Please generate an expanded query containing synonyms, relevant entities, and keywords based on the user's original query to improve retrieval recall. Output ONLY the expanded keyword string in Simplified Chinese without any explanation or prefix.\n\nOriginal Query: {query_text}\nExpanded Query:"
    else:
        prompt = f"You are a search query optimizer. Please generate an expanded query containing synonyms, relevant entities, and keywords based on the user's original query to improve retrieval recall. Output ONLY the expanded keyword string without any explanation or prefix.\n\nOriginal Query: {query_text}\nExpanded Query:"
    
    return llm_generate(prompt)


def rerank_chunks(query, chunks, language, top_k=5):
    """
    Reranks the retrieved chunks using LLM and returns the top_k most relevant chunks.
    """
    if not chunks:
        return []
        
    # Prepare the prompt for reranking
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        # Use a simplified representation to save tokens
        content_preview = chunk['page_content'][:200].replace("\n", " ")
        chunks_text += f"[{i}] {content_preview}...\n"
        
    if language == 'zh':
        prompt = f"""
You are a precise document reranking assistant. Please select the top {top_k} most relevant document chunks from the candidates below based on the user's query.
Output ONLY the list of indices of the most relevant chunks in a JSON array format, e.g., [0, 2, 1]. Order them by relevance from high to low.

Query: {query}

Candidate Document Chunks:
{chunks_text}

Output (JSON Array ONLY):
"""
    else:
        prompt = f"""
You are a precise document reranking assistant. Please select the top {top_k} most relevant document chunks from the candidates below based on the user's query.
Output ONLY the list of indices of the most relevant chunks in a JSON array format, e.g., [0, 2, 1]. Order them by relevance from high to low.

Query: {query}

Candidate Document Chunks:
{chunks_text}

Output (JSON Array ONLY):
"""

    response = llm_generate(prompt)
    
    # Parse the response to get indices
    try:
        # Clean up response to ensure it's a valid list
        import re
        import json
        
        # Find the first '[' and last ']'
        start = response.find('[')
        end = response.rfind(']')
        
        if start != -1 and end != -1:
            json_str = response[start:end+1]
            indices = json.loads(json_str)
            
            # Filter valid indices
            valid_indices = [idx for idx in indices if isinstance(idx, int) and 0 <= idx < len(chunks)]
            
            # Get the actual chunks
            reranked_chunks = [chunks[idx] for idx in valid_indices]
            
            # If we got fewer than top_k, fill with remaining chunks in original order
            if len(reranked_chunks) < top_k:
                seen_indices = set(valid_indices)
                for i in range(len(chunks)):
                    if i not in seen_indices:
                        reranked_chunks.append(chunks[i])
                        if len(reranked_chunks) >= top_k:
                            break
                            
            return reranked_chunks[:top_k]
        else:
            print(f"Warning: Could not parse reranking response: {response}")
            return chunks[:top_k]
            
    except Exception as e:
        print(f"Error parsing reranking response: {e}")
        return chunks[:top_k]
