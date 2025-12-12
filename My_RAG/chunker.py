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
