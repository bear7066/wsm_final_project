from typing import List, Dict
import requests
import numpy as np
import os

try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False


class RemoteFlagReranker:
    """
    Fake FlagReranker class: same interface as the official one (More information can be found in FlagEmbedding; https://github.com/FlagOpen/FlagEmbedding),
    but internally calls a remote API.
    """

    def __init__(self, api_url: str):
        """
        api_url: your rerank endpoint
        """
        self.api_url = api_url

    def compute_score(self, pairs, max_length=1024):
        """
        pairs: list of [text1, text2], same as the official compute_score

        return: score of each pair in np.ndarray, same as the official compute_score
        """
        payload = {"pairs": [{"text1": a, "text2": b} for a, b in pairs]}

        resp = requests.post(self.api_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"API request failed ({resp.status_code}): {resp.text}")

        scores = resp.json()["scores"]
        return np.array(scores)


class BGEReranker:
    def __init__(self, model_name_or_url: str = None, use_remote: bool = False):
        self.use_remote = use_remote
        
        # Default logic for model/url
        if not model_name_or_url:
            if self.use_remote:
                # Default remote URL
                model_name_or_url = "http://ollama-gateway:11434/rerank"
            else:
                # Default local model
                model_name_or_url = "BAAI/bge-reranker-v2-m3"

        if self.use_remote:
            print(f"Initializing RemoteFlagReranker with URL: {model_name_or_url}")
            self.reranker = RemoteFlagReranker(model_name_or_url)
        else:
            print(f"Initializing Local FlagReranker with model: {model_name_or_url}")
            if not FLAG_EMBEDDING_AVAILABLE:
                raise ImportError("FlagEmbedding is not installed. Please install it or use remote mode.")
            self.reranker = FlagReranker(model_name_or_url, use_fp16=True)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        if not chunks:
            return []
            
        # pairs = [[query, chunk['page_content']] for chunk in chunks]
        # Use first 2000 chars of chunk content to avoid excessive length, 
        # though reranker truncates at 1024 tokens anyway.
        pairs = [[query, chunk['page_content']] for chunk in chunks]

        try:
            if self.use_remote:
                # Handle batching (limit 32 pairs per call)
                all_scores = []
                batch_size = 32
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    try:
                        scores = self.reranker.compute_score(batch, max_length=1024)
                        # Ensure scores is a flat array (compute_score might return list of lists for 1 pair?)
                        # FlagReranker returns list of float or list of list? 
                        # FlagReranker.compute_score returns np.ndarray.
                        if isinstance(scores, list):
                            all_scores.extend(scores)
                        elif isinstance(scores, np.ndarray):
                            all_scores.extend(scores.tolist())
                        else:
                            all_scores.extend([scores])
                    except Exception as e:
                        print(f"Error reranking batch {i // batch_size}: {e}")
                        # Fallback for failed batch: append -infinity scores
                        all_scores.extend([-9999.0] * len(batch))
                
                final_scores = np.array(all_scores)
            else:
                final_scores = self.reranker.compute_score(pairs, max_length=1024)
                if not isinstance(final_scores, np.ndarray):
                    final_scores = np.array(final_scores)

            # Sort chunks by score
            # argsort sorts ascending, so we reverse it
            sorted_indices = np.argsort(final_scores)[::-1]
            
            top_indices = sorted_indices[:top_k]
            results = [chunks[i] for i in top_indices]
            
            return results

        except Exception as e:
            print(f"Reranking failed: {e}. Returning original chunks.")
            return chunks[:top_k]


def create_reranker(reranker_type="bge"):
    # Check env var to force remote mode
    # 
    use_remote = True
    # if os.environ.get("USE_REMOTE_RERANKER") == "true":
    #     print("USE_REMOTE_RERANKER is/set to true. Using Remote Reranker.")
    #     use_remote = True
    # elif not FLAG_EMBEDDING_AVAILABLE:
    #     print("FlagEmbedding not found. Switching to Remote Reranker.")
    #     use_remote = True
    # else:
    #     use_remote = False
    
    return BGEReranker(use_remote=use_remote)

if __name__ == "__main__":
    # Test
    reranker = create_reranker()
    pairs = [
        {"page_content": "machine learning is a field of study..."},
        {"page_content": "Paris is the capital of France."}
    ]
    results = reranker.rerank("what is machine learning?", pairs, top_k=1)
    print([r['page_content'] for r in results])