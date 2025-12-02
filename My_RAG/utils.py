import jsonlines
from ollama import Client

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
        client = Client()
        response = client.generate(model=model, prompt=prompt, stream=False)
        return response.get("response", "No response from model.")
    except Exception as e:
        return f"Error using Ollama Python client: {e}"