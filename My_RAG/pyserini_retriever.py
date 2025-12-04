from pyserini.search.lucene import LuceneSearcher
import json
import os
import tempfile
import shutil
from pathlib import Path


class PyseriniRetriever:
    
    def __init__(self, chunks, language="en", index_dir=None, keep_index=True):
        """
        Initialize Pyserini retriever.
        
        Args:
            chunks: List of document chunks with 'page_content' field
            language: Language code ('en' or 'zh')
            index_dir: Optional path to save/load index. If None, uses temp directory
            keep_index: Whether to keep index after retriever is destroyed
        """
        self.chunks = chunks
        self.language = language
        self.keep_index = keep_index
        
        # Set up index directory
        if index_dir:
            self.index_dir = index_dir
            os.makedirs(self.index_dir, exist_ok=True)
        else:
            self.index_dir = tempfile.mkdtemp(prefix="pyserini_index_")
        
        print(f"Index directory: {self.index_dir}")
        
        # Build index if not exists
        if not self._index_exists():
            print("Building Pyserini index...")
            self._build_index()
            print("Index built successfully.")
        else:
            print("Loading existing index...")
        
        # Initialize searcher
        self.searcher = LuceneSearcher(self.index_dir)
        
        # Configure BM25 parameters (Standard baseline)
        # k1=1.2, b=0.75 are generally robust defaults
        self.searcher.set_bm25(k1=1.2, b=0.75)

        # Enable RM3 Query Expansion (Improves recall)
        # fb_terms=10: number of expansion terms
        # fb_docs=10: number of expansion documents
        # original_query_weight=0.5: weight of original query
        self.searcher.set_rm3(fb_terms=10, fb_docs=10, original_query_weight=0.5)        
        # Set language for analyzer (important for Chinese)
        if language == "zh":
            # Pyserini uses CJK analyzer for Chinese
            self.searcher.set_language('zh')
        
        print(f"Retriever initialized with {len(chunks)} chunks.")
    
    def _index_exists(self):
        """Check if index already exists"""
        # Check for key Lucene index files
        required_files = ['segments_1']  # Lucene index marker file
        return all((Path(self.index_dir) / f).exists() for f in required_files)
    
    def _build_index(self):
        """Build Pyserini index from chunks"""
        # Create temporary collection directory
        collection_dir = tempfile.mkdtemp(prefix="pyserini_collection_")
        
        try:
            # Write documents in JSONL format
            docs_file = os.path.join(collection_dir, 'documents.jsonl')
            with open(docs_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(self.chunks):
                    doc = {
                        'id': str(i),
                        'contents': chunk['page_content'],
                        # Store metadata if needed
                        'metadata': json.dumps(chunk.get('metadata', {}), ensure_ascii=False)
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            # Build index using pyserini
            from pyserini.index.lucene import LuceneIndexer
            
            indexer_args = [
                '--collection', 'JsonCollection',
                '--input', collection_dir,
                '--index', self.index_dir,
                '--generator', 'DefaultLuceneDocumentGenerator',
                '--threads', '1',
                '--storePositions',
                '--storeDocvectors',
                '--storeRaw'
            ]
            
            # Add language-specific settings
            if self.language == 'zh':
                indexer_args.extend(['--language', 'zh'])
            
            # Run indexer
            import sys
            from io import StringIO
            
            # Capture stdout to reduce verbosity
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Use pyserini's command-line interface
                import subprocess
                # Use sys.executable to get the current Python interpreter (works in venv)
                cmd = [sys.executable, '-m', 'pyserini.index.lucene'] + indexer_args
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    # Restore stdout and show error
                    sys.stdout = old_stdout
                    print(f"Indexing stderr: {result.stderr}")
                    raise RuntimeError(f"Indexing failed: {result.stderr}")
            finally:
                sys.stdout = old_stdout
        
        finally:
            # Clean up collection directory
            shutil.rmtree(collection_dir, ignore_errors=True)
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries
        """
        # Search using Pyserini
        hits = self.searcher.search(query, k=top_k)
        
        # Convert hits back to original chunk format
        results = []
        for hit in hits:
            doc_id = int(hit.docid)
            if doc_id < len(self.chunks):
                results.append(self.chunks[doc_id])
        
        return results
    
    def retrieve_with_scores(self, query, top_k=5):
        """
        Retrieve chunks with their BM25 scores.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of tuples (chunk, score)
        """
        hits = self.searcher.search(query, k=top_k)
        
        results = []
        for hit in hits:
            doc_id = int(hit.docid)
            if doc_id < len(self.chunks):
                results.append((self.chunks[doc_id], hit.score))
        
        return results
    
    def __del__(self):
        """Cleanup index directory if not keeping"""
        if not self.keep_index and hasattr(self, 'index_dir'):
            if os.path.exists(self.index_dir):
                shutil.rmtree(self.index_dir, ignore_errors=True)


def create_retriever(chunks, language, index_dir=None, keep_index=True):
    """
    Creates a Pyserini retriever from document chunks.
    
    Args:
        chunks: List of document chunks
        language: Language code ('en' or 'zh')
        index_dir: Optional directory to save index (for reuse)
        keep_index: Whether to persist index after program ends
        
    Returns:
        PyseriniRetriever instance
    """
    return PyseriniRetriever(chunks, language, index_dir=index_dir, keep_index=keep_index)
