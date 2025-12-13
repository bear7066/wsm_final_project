from bm25 import BM25Retriever
from gemma_retriever import VectorRetriever
from typing import List, Dict


class HybridRetriever:
    """Combines BM25 and Vector search with weighted sum"""

    def __init__(self, chunks, language="en", bm25_weight=0.6, vector_weight=0.4, embedding_model="embeddinggemma:300m"):
        """
        Initialize hybrid retriever.
        
        Args:
            chunks: List of document chunks
            language: 'en' or 'zh'
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            embedding_model: Embedding model name (auto-select if None)
        """
        self.chunks = chunks
        self.language = language
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        print(f"Initializing Hybrid Retriever (BM25: {bm25_weight}, Vector: {vector_weight})")

        print("\n1️⃣  Initializing BM25 Retriever...")
        self.bm25_retriever = BM25Retriever(chunks, language)

        print("\n2️⃣  Initializing Vector Retriever...")
        self.vector_retriever = VectorRetriever(chunks, language, embedding_model)

        print("\n✅ Hybrid Retriever ready!")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization to 0-1"""
        if not scores or len(scores) == 1:
            return [1.0] * len(scores)

        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = None) -> List[Dict]:
        """
        Hybrid retrieval with score fusion.
        
        Args:
            query: Query string
            top_k: Number of final results
            candidate_k: Number of candidates from each retriever (default: top_k * 2)
            
        Returns:
            List of merged and ranked chunks
        """
        if candidate_k is None:
            candidate_k = top_k * 2

        # Step 1: Get BM25 results
        bm25_results_with_scores = self.bm25_retriever.retrieve_with_scores(query, top_k=candidate_k)
        
        bm25_chunk_scores = {
            self.chunks.index(chunk): score 
            for chunk, score in bm25_results_with_scores
        }

        # Step 2: Get Vector results
        vector_results = self.vector_retriever.retrieve_with_scores(query, top_k=candidate_k)
        vector_chunk_scores = {
            self.chunks.index(chunk): score 
            for chunk, score in vector_results
        }

        # Step 3: Merge all unique chunks
        all_indices = set(bm25_chunk_scores.keys()) | set(vector_chunk_scores.keys())

        # Step 4: Normalize scores
        bm25_list = [bm25_chunk_scores.get(idx, 0) for idx in all_indices]
        vector_list = [vector_chunk_scores.get(idx, 0) for idx in all_indices]

        norm_bm25 = self._normalize_scores(bm25_list)
        norm_vector = self._normalize_scores(vector_list)

        # Step 5: Weighted sum
        combined = {
            idx: self.bm25_weight * norm_bm25[i] + self.vector_weight * norm_vector[i] 
            for i, idx in enumerate(all_indices)
        }

        # Step 6: Sort and return top-k
        sorted_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [self.chunks[idx] for idx, _ in sorted_indices[:top_k]]


def create_retriever(chunks, language, bm25_weight=None, vector_weight=None, embedding_model=None):
    """
    Create hybrid retriever with language-specific weights.
    
    Args:
        chunks: List of document chunks
        language: 'en' or 'zh'
        bm25_weight: Weight for BM25 (override default)
        vector_weight: Weight for vector (override default)
        embedding_model: Embedding model name (auto-select if None)
        
    Returns:
        HybridRetriever instance
    """
    if bm25_weight is None or vector_weight is None:
        if language == "zh":
            # zh -> 0.6 0.4 good
            bm25_weight = 0.6
            vector_weight = 0.4
        else:
            # en -> 0.5 0.5 good
            bm25_weight = 0.5
            vector_weight = 0.5
            
    # 
    if embedding_model is None:
        if language == "zh":
            embedding_model = "qwen3-embedding:0.6b"
        else:
            embedding_model = "embeddinggemma:300m"

    print(f"Creating Hybrid Retriever for {language} with weights - BM25: {bm25_weight}, Vector: {vector_weight}")
    print(f"Using Embedding Model: {embedding_model}")
    return HybridRetriever(chunks, language, bm25_weight, vector_weight, embedding_model)
