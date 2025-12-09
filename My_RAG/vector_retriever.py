"""
Vector Retriever: Dense retrieval using Gemma embeddings
"""

from ollama import Client
from utils import load_ollama_config
import numpy as np
from typing import List, Dict, Tuple


class VectorRetriever:
    """Dense retriever using Gemma embeddings"""
    
    def __init__(self, chunks, language="en", model_name="gemma:2b"):
        self.chunks = chunks
        self.model_name = model_name
        
        # Setup Ollama client
        config = load_ollama_config()
        self.client = Client(host=config["host"])
        
        print(f"Building Gemma embeddings for {len(chunks)} chunks...")
        
        # Build embeddings
        corpus = [chunk['page_content'] for chunk in chunks]
        self.embeddings = np.array([self._get_embedding(text) for text in corpus])
        
        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Gemma"""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            return np.array(response['embedding'])
        except:
            return np.zeros(2048)  # Fallback
    
    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve with scores"""
        query_emb = self._get_embedding(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve without scores"""
        return [chunk for chunk, _ in self.retrieve_with_scores(query, top_k)]


def create_retriever(chunks, language, model_name="gemma:2b"):
    """Create vector retriever"""
    return VectorRetriever(chunks, language, model_name)
