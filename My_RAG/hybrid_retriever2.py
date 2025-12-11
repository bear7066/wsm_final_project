"""
Hybrid Retriever: Combines BM25 + Vector Embeddings
RRF integration
"""

from retriever import BM25Retriever
from vector_retriever import VectorRetriever
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

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = None) -> List[Dict]:
        """
        Hybrid retrieval with Weighted Reciprocal Rank Fusion (RRF).
        
        Args:
            query: Query string
            top_k: Number of final results
            candidate_k: Number of candidates from each retriever (default: top_k * 2)
            
        Returns:
            List of merged and ranked chunks
        """
        if candidate_k is None:
            candidate_k = top_k * 2

        # Step 1: Get results from both retrievers
        # We use retrieve_with_scores for consistency, but for RRF we mostly care about the RANK (order)
        bm25_results = self.bm25_retriever.retrieve_with_scores(query, top_k=candidate_k)
        vector_results = self.vector_retriever.retrieve_with_scores(query, top_k=candidate_k)

        # Step 2: Calculate Weighted RRF
        # Score = weight * (1 / (rank + k))
        # k is a smoothing constant, typically 60
        rrf_k = 60
        fused_scores = {}

        # Process BM25 results
        for rank, (chunk, _) in enumerate(bm25_results):
            chunk_idx = self.chunks.index(chunk)
            if chunk_idx not in fused_scores:
                fused_scores[chunk_idx] = 0.0
            fused_scores[chunk_idx] += self.bm25_weight * (1 / (rank + rrf_k))

        # Process Vector results
        for rank, (chunk, _) in enumerate(vector_results):
            chunk_idx = self.chunks.index(chunk)
            if chunk_idx not in fused_scores:
                fused_scores[chunk_idx] = 0.0
            fused_scores[chunk_idx] += self.vector_weight * (1 / (rank + rrf_k))

        # Step 3: Sort and return top-k
        sorted_indices = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.chunks[idx] for idx, _ in sorted_indices[:top_k]]


def create_retriever(chunks, language, bm25_weight=0.6, vector_weight=0.4, embedding_model="embeddinggemma:300m"):
    """
    Create hybrid retriever.
    Args:
        chunks: List of document chunks
        language: 'en' or 'zh'
        bm25_weight: Weight for BM25 (default: 0.7)
        vector_weight: Weight for vector (default: 0.3)
        embedding_model: Embedding model name (auto-select if None)
        
    Returns:
        HybridRetriever instance
    """
    print("RRF hybrid retriever")
    return HybridRetriever(chunks, language, bm25_weight, vector_weight, embedding_model)


if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock, patch

    class TestHybridRetriever(unittest.TestCase):
        def setUp(self):
            self.chunks = [
                {"page_content": "Chunk A: Apple", "metadata": {"id": 0}},
                {"page_content": "Chunk B: Banana", "metadata": {"id": 1}},
                {"page_content": "Chunk C: Cherry", "metadata": {"id": 2}},
            ]
            self.query = "fruit"

        @patch('hybrid_retriever.BM25Retriever')
        @patch('hybrid_retriever.VectorRetriever')
        def test_hybrid_retrieve_ranking(self, MockVectorRetriever, MockBM25Retriever):
            # Setup Mocks
            mock_bm25_instance = MockBM25Retriever.return_value
            mock_vector_instance = MockVectorRetriever.return_value
            
            # Scenario:
            # BM25 favors Chunk A (0) heavily.
            # Vector favors Chunk C (2) heavily.
            # Chunk B (1) is mediocre in both.
            
            # BM25 retrieve returns list of chunks
            # BM25 retrieve_with_scores returns list of (chunk, score)
            mock_bm25_instance.retrieve_with_scores.return_value = [
                (self.chunks[0], 10.0),
                (self.chunks[1], 5.0)
            ]
            
            # Vector retrieve_with_scores returns list of (chunk, score)
            mock_vector_instance.retrieve_with_scores.return_value = [
                (self.chunks[2], 0.9),
                (self.chunks[1], 0.8)
            ]

            # Case 1: High BM25 weight (0.7 vs 0.3)
            retriever = HybridRetriever(self.chunks, bm25_weight=0.7, vector_weight=0.3)
            # Inject mocks
            retriever.bm25_retriever = mock_bm25_instance
            retriever.vector_retriever = mock_vector_instance
            
            results = retriever.retrieve(self.query, top_k=3)
            
            print("\n[Test Case 1] Weights: BM25=0.7, Vector=0.3")
            for r in results:
                print(f" - {r['page_content']}")
                
            self.assertEqual(results[0]['metadata']['id'], 1, "Chunk B should be first due to RRF consensus (appearing in both)")
            self.assertEqual(results[1]['metadata']['id'], 0, "Chunk A should be second (High BM25 weight)")
            self.assertEqual(results[2]['metadata']['id'], 2)

        @patch('hybrid_retriever.BM25Retriever')
        @patch('hybrid_retriever.VectorRetriever')
        def test_vector_dominance(self, MockVectorRetriever, MockBM25Retriever):
            # Same scores, but Vector weight is high
            mock_bm25_instance = MockBM25Retriever.return_value
            mock_vector_instance = MockVectorRetriever.return_value
            
            mock_bm25_instance.retrieve_with_scores.return_value = [
                (self.chunks[0], 10.0),
                (self.chunks[1], 5.0)
            ] 
            
            mock_vector_instance.retrieve_with_scores.return_value = [
                (self.chunks[2], 0.9),
                (self.chunks[1], 0.8)
            ]
            
            # Weights: BM25=0.2, Vector=0.8
            retriever = HybridRetriever(self.chunks, bm25_weight=0.2, vector_weight=0.8)
            retriever.bm25_retriever = mock_bm25_instance
            retriever.vector_retriever = mock_vector_instance
            
            results = retriever.retrieve(self.query, top_k=3)
            
            print("\n[Test Case 2] Weights: BM25=0.2, Vector=0.8")
            for r in results:
                print(f" - {r['page_content']}")
            
            # With B=0.81 and C=0.8 and A=0.2, B should be first.
            self.assertEqual(results[0]['metadata']['id'], 1, "Chunk B should be first due to RRF consensus")
            self.assertEqual(results[1]['metadata']['id'], 2, "Chunk C should be second (High Vector weight)")
            self.assertEqual(results[2]['metadata']['id'], 0)

    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)