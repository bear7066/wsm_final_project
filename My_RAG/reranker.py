from typing import List, Dict
from utils import load_ollama_config
from ollama import Client
import json

class OllamaReranker:
    def __init__(self, model_name: str = None):
        """
        Initialize the Ollama Reranker.
        Args:
            model_name: Optional override for the model name. If None, loads from config.
        """
        print(f"Loading Ollama Reranker...")
        try:
            config = load_ollama_config()
            self.host = config.get("host")
            # prioritize argument, then explicitly default to bge-reranker-v2-m3
            # We ignore config.get("model") because that is usually the LLM (granite), not the reranker.
            self.model_name = model_name if model_name else "qllama/bge-reranker-v2-m3:latest"
            
            self.client = Client(host=self.host)
            print(f"Ollama Reranker connected to {self.host} using model {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load Ollama config, defaulting to localhost. Error: {e}")
            self.client = Client()
            self.model_name = model_name if model_name else "qllama/bge-reranker-v2-m3:latest"

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Reranks a list of chunks based on the query using Ollama (LLM-based).
        """
        if not chunks:
            return []

        # Detect language
        language = 'zh' if self._is_chinese(query) else 'en'
        
        # Prepare chunks text
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            # Use a simplified representation to save tokens
            content_preview = chunk['page_content'][:150].replace("\n", " ")
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

        try:
            response = self.client.generate(
                model=self.model_name, 
                prompt=prompt, 
                stream=False, 
                options={
                    "temperature": 0.0,
                    "num_ctx": 16384  # Increase context window to handle large reranking prompts
                }
            )
            response_text = response.get("response", "")
            
            # Parse the response to get indices
            # Find the first '[' and last ']'
            start = response_text.find('[')
            end = response_text.rfind(']')
            
            if start != -1 and end != -1:
                json_str = response_text[start:end+1]
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
                print(f"Warning: Could not parse reranking response: {response_text}")
                return chunks[:top_k]
                
        except Exception as e:
            print(f"Error during reranking: {e}")
            return chunks[:top_k]

def create_reranker(reranker_type="ollama"):
    if reranker_type == "ollama":
        return OllamaReranker()
    else:
        return BGEReranker()

if __name__ == "__main__":
    # Simple test
    reranker = create_reranker()
    query = "What is the capital of France?"
    chunks = [
        {"page_content": "The Eiffel Tower is in Paris."},
        {"page_content": "London is the capital of the UK."},
        {"page_content": "Paris is the capital of France.", "metadata": {"id": 1}},
        {"page_content": "France is a country in Europe."},
    ]
    
    print("\nReranking...")
    results = reranker.rerank(query, chunks, top_k=2)
    for r in results:
        print(f"- {r['page_content']}")
