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
    
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    
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
=======
    answer = llm_generate(query, context_chunks)
>>>>>>> Stashed changes
    print("Generated Answer:", answer)