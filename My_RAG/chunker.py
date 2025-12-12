def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str):
            text = doc['content']
            # Use recursive chunking for ALL languages
            text_chunks = recursive_split(text, chunk_size, chunk_overlap)
            
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


def recursive_split(text, chunk_size, chunk_overlap):
    separators = ["\n\n", "\n", " ", ""]
    
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

