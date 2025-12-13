
import sys
import os

# Add My_RAG to path
sys.path.append(os.path.join(os.getcwd(), 'My_RAG'))

from utils import load_ollama_config
from hybrid_retriever import create_retriever

def check_config():
    print("Checking Ollama Configuration...")
    try:
        config = load_ollama_config()
        print(f"Config loaded successfully.")
        print(f"Host: {config.get('host')}")
        print(f"Model (for Generation): {config.get('model')}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def check_retriever_defaults():
    print("\nChecking Retriever Defaults...")
    # Inspect default args of create_retriever
    import inspect
    sig = inspect.signature(create_retriever)
    embedding_model = sig.parameters['embedding_model'].default
    print(f"HybridRetriever default embedding model: {embedding_model}")

if __name__ == "__main__":
    check_config()
    check_retriever_defaults()
