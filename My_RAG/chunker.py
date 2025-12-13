def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            text_len = len(text)
            lang = doc['language']
            start_index = 0
            chunk_count = 0
            if lang == language:
                while start_index < text_len:
                    end_index = min(start_index + chunk_size, text_len)
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = chunk_count
                    chunk = {
                        'page_content': text[start_index:end_index],
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
                    start_index += chunk_size - chunk_overlap
                    chunk_count += 1
    return chunks
# recursive_return_one_chunk.json

def recursive_chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str):
            text = doc['content']
            # Use recursive chunking for ALL languages, pass language param
            text_chunks = _recursive_split(text, chunk_size, chunk_overlap, language=language)
            
            for i, text_chunk in enumerate(text_chunks):
                chunk_metadata = doc.copy()
                chunk_metadata.pop('content', None)
                chunk_metadata['chunk_index'] = i
                chunk = {
                    'page_content': text_chunk,
                    'metadata': chunk_metadata
                }
                chunks.append(chunk)
    return chunks


def _recursive_split(text, chunk_size, chunk_overlap, language="en"):
    if language == "zh":
        separators = ["\n\n", "\n", "。", "！", "？", " ", ""]
    else:
        separators = ["\n\n", "\n", "。", "！", "？", " ", ""]
        # Optimized separators for English as requested
        # separators = [
        #     "\n\n",  # 1. Paragraphs
        #     "\n",    # 2. Lines
        #     ". ",    # 3. Sentences (Dot + Space)
        #     "? ",    # 4. Questions
        #     "! ",    # 5. Exclamations
        #     "; ",    # 6. Semicolons
        #     " ",     # 7. Words
        #     ""       # 8. Characters
        # ]
    
    def _split_text_recursive(text, separators):
        # Determine which separator to use
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break
        
        # Split
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
            
        # Process splits with recursion
        good_splits = []
        for s in splits:
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    good_splits.extend(_split_text_recursive(s, new_separators))
                else:
                    # Hard truncate if no more separators
                    for i in range(0, len(s), chunk_size - chunk_overlap):
                        good_splits.append(s[i:i+chunk_size])
        
        # Merge splits into chunks
        return _merge_splits(good_splits, separator, chunk_size, chunk_overlap)

    def _merge_splits(splits, separator, chunk_size, chunk_overlap):
        docs = []
        current_doc = []
        total_len = 0
        sep_len = len(separator)
        
        for split in splits:
            split_len = len(split)
            
            # If adding this split exceeds chunk_size, we need to finalize the current chunk
            if total_len + split_len + (sep_len if current_doc else 0) > chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    
                    # Manage overlap: remove from front until we fit within overlap/size constraints
                    # 1. Reduce to overlap size
                    while total_len > chunk_overlap and current_doc:
                        p = current_doc.pop(0)
                        total_len -= len(p) + (sep_len if current_doc else 0)
                    
                    # 2. Further reduce if adding the NEW split would still exceed chunk_size
                    # (This prevents created chunks from exceeding chunk_size)
                    while total_len + split_len + (sep_len if current_doc else 0) > chunk_size and current_doc:
                        p = current_doc.pop(0)
                        total_len -= len(p) + (sep_len if current_doc else 0)
            
            current_doc.append(split)
            total_len += split_len + (sep_len if len(current_doc) > 1 else 0)
            
        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
                
        return docs

    return _split_text_recursive(text, separators)

