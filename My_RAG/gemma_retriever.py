import numpy as np
import ollama
from ollama import Client
from utils import load_ollama_config

class VectorRetriever:
    def __init__(self, chunks, language="en", model_name="gemma:300m"):
        """
        Initialize the Vector Retriever using Ollama embeddings.
        Args:
            chunks: List of document chunks.
            language: Language of the chunks.
            model_name: The Ollama model to use for embeddings.
        """
        self.chunks = chunks
        self.language = language
        self.model_name = model_name
        self.embeddings = []
        
        # Load config and initialize client
        try:
            config = load_ollama_config()
            self.client = Client(host=config["host"])
            print(f"VectorRetriever connected to Ollama at {config['host']}")
        except Exception as e:
            print(f"Warning: Could not load Ollama config, defaulting to localhost. Error: {e}")
            self.client = Client()

        print(f"Loading Vector Retriever with model: {self.model_name}")
        self._build_index()

    def _get_embedding(self, text):
        """Generates embedding for a single text using Ollama."""
        try:
            # Use the initialized client
            response = self.client.embeddings(model=self.model_name, prompt=text)
            return response.get("embedding")
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def _build_index(self):
        """Pre-computes embeddings for all chunks."""
        print(f"Computing embeddings for {len(self.chunks)} chunks...")
        self.embeddings = []
        for i, chunk in enumerate(self.chunks):
            text = chunk.get("page_content", "")
            emb = self._get_embedding(text)
            if emb:
                self.embeddings.append(emb)
            else:
                # Handle failure (e.g., use zero vector or skip)
                self.embeddings.append([0.0] * 768) # Placeholder size, adjust if known
        self.embeddings = np.array(self.embeddings)
        print("Embeddings computed.")

    def retrieve_with_scores(self, query, top_k=5):
        """
        Retrieves top_k chunks based on cosine similarity.
        Returns list of (chunk, score).
        """
        query_emb = self._get_embedding(query)
        if not query_emb:
            return []

        query_vec = np.array(query_emb)
        
        # Calculate Cosine Similarity
        # Sim(A, B) = (A . B) / (||A|| * ||B||)
        
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return []

        scores = []
        for idx, doc_vec in enumerate(self.embeddings):
            norm_d = np.linalg.norm(doc_vec)
            if norm_d == 0:
                score = 0
            else:
                score = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
            scores.append((idx, score))
        
        # Sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.chunks[idx], float(score)))
            
        return results

    def retrieve(self, query, top_k=5):
        """Wrapper to return just chunks."""
        results_with_scores = self.retrieve_with_scores(query, top_k)
        return [r[0] for r in results_with_scores]