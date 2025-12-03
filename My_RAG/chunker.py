# Modification: Implemented Sentence-aware Chunking
# 1. Added split_sentences function to split text into sentences based on punctuation.
# 2. Updated chunk_documents to build chunks from complete sentences to preserve semantic integrity.
# 3. Implemented sliding window with overlap based on character count.

import re

def split_sentences(text, language):
    """Splits text into sentences based on language-specific punctuation."""
    if language == 'zh':
        # Split by Chinese punctuation (。, ？, ！, …) and keep the punctuation
        # Added ； (semicolon) as it often separates complete thoughts in lists
        parts = re.split(r'([。？！…；])', text)
        sentences = []
        # Recombine content with its punctuation
        for i in range(0, len(parts) - 1, 2):
            sentences.append(parts[i] + parts[i+1])
        # Add the last part if it exists (e.g. text without ending punctuation)
        if len(parts) % 2 == 1 and parts[-1]:
            sentences.append(parts[-1])
        return sentences
    else:
        # Split by English punctuation (. ? !) followed by whitespace
        # Lookbehind assertion ensures punctuation is kept with the sentence
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s for s in sentences if s.strip()]

def chunk_documents(docs, language, chunk_size=500, chunk_overlap=150):
    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' not in doc or not isinstance(doc['content'], str) or 'language' not in doc:
            continue
            
        text = doc['content']
        lang = doc['language']
        
        # Only process documents of the target language
        if lang != language:
            continue
            
        sentences = split_sentences(text, lang)
        
        current_chunk_sentences = []
        current_chunk_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Check if adding this sentence would exceed the chunk size
            if current_chunk_len + sentence_len > chunk_size and current_chunk_sentences:
                # 1. Save the current chunk
                chunk_text = "".join(current_chunk_sentences)
                chunk_metadata = doc.copy()
                chunk_metadata.pop('content', None)
                chunk_metadata['chunk_index'] = len(chunks)
                
                chunks.append({
                    'page_content': chunk_text,
                    'metadata': chunk_metadata
                })
                
                # 2. Handle Overlap: Keep sentences from the end that fit within chunk_overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                # 3. Start new chunk with overlap + current sentence
                current_chunk_sentences = overlap_sentences
                current_chunk_sentences.append(sentence)
                current_chunk_len = overlap_len + sentence_len
                
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_len += sentence_len
        
        # Add the last chunk if it has content
        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences)
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)
            chunk_metadata['chunk_index'] = len(chunks)
            chunks.append({
                'page_content': chunk_text,
                'metadata': chunk_metadata
            })
            
    return chunks
