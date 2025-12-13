
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """
        Initialize the Reranker with a Cross-Encoder model.
        Args:
            model_name: The name of the Cross-Encoder model to use.
                        Recommended: 'BAAI/bge-reranker-v2-m3' (Multilingual)
                                     'cross-encoder/ms-marco-MiniLM-L-6-v2' (English, fast)
        """
        print(f"Loading Reranker model: {model_name}...")
        try:
            self.model = CrossEncoder(model_name)
            print("Reranker model loaded successfully.")
        except Exception as e:
            print(f"Error loading Reranker model: {e}")
            self.model = None

    def rerank(self, query, docs, top_k=5):
        """
        Rerank a list of documents based on the query.
        
        Args:
            query: The search query string.
            docs: A list of document chunks (dicts with 'page_content').
            top_k: The number of documents to return after reranking.
            
        Returns:
            A list of reranked document chunks (top_k).
        """
        if not self.model or not docs:
            return docs[:top_k]

        # Prepare pairs for Cross-Encoder: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc['page_content']] for doc in docs]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Combine docs with scores
        doc_scores = list(zip(docs, scores))
        
        # Sort by score in descending order
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Return top_k docs
        return [doc for doc, score in sorted_doc_scores[:top_k]]

def create_reranker(model_name="BAAI/bge-reranker-v2-m3"):
    return Reranker(model_name)
