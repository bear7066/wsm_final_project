import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict

class BGEReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = None):
        """
        Initializes the BGE Reranker (Cross-Encoder).
        
        Args:
            model_name: HuggingFace model ID.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        print(f"Loading Reranker model '{model_name}' on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Reranker model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading reranker model: {e}")
            raise e

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Reranks a list of chunks based on the query using the Cross-Encoder.
        
        Args:
            query: The search query.
            chunks: List of document chunks (must contain 'page_content').
            top_k: Number of chunks to return after reranking.
            
        Returns:
            Top-k reranked chunks.
        """
        if not chunks:
            return []
            
        # Preparing pairs for the model: [[query, doc1], [query, doc2], ...]
        pairs = [[query, chunk['page_content']] for chunk in chunks]
        
        with torch.no_grad():
            # tokenize
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )
            
            # move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # inference
            # BGE-Reranker outputs logits. Higher logit = more relevant.
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
            
        # Convert to numpy/list
        scores = scores.cpu().numpy().tolist()
        
        # Combine chunks with scores
        scored_chunks = []
        for i, score in enumerate(scores):
            scored_chunks.append((chunks[i], score))
            
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Print top 3 scores
        # print(f"Top 3 Rerank Scores: {[round(s, 4) for _, s in scored_chunks[:3]]}")
        
        # Return top_k chunks (stripping scores for compatibility)
        return [chunk for chunk, _ in scored_chunks[:top_k]]

def create_reranker(model_name="BAAI/bge-reranker-v2-m3"):
    return BGEReranker(model_name)

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
