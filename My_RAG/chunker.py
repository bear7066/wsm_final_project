from ollama import Client
from utils import load_ollama_config

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
                chunk_metadata['original_content'] = text_chunk  # Save original clean text
                
                # Generate context
                context = _generate_chunk_context(language, text, text_chunk, chunk_metadata)
                
                chunk = {
                    'page_content': context + "\n" + text_chunk,  # Prepend context
                    'metadata': chunk_metadata
                }
                chunks.append(chunk)
    return chunks

def _generate_chunk_context(language, doc_text, chunk_text, metadata=None):
    """
    Generate contextual description for a specific chunk using Ollama.
    This follows Anthropic's Contextual Retrieval approach.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # Truncate doc_text to avoid token limits (keep first 6000 chars for smaller models)
    doc_text_truncated = doc_text[:6000]
    
    # Extract subject name from metadata
    subject_name = ""
    if metadata:
        if "company_name" in metadata:
            subject_name = metadata["company_name"]
        elif "hospital_patient_name" in metadata:
            subject_name = metadata["hospital_patient_name"]
        elif "court_name" in metadata:
            subject_name = metadata["court_name"]
    
    if language == "zh":
        subject_instruction = ""
        if subject_name:
            subject_instruction = f'\n重要：本文档的主体是「{subject_name}」。'
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

以下是我们想要在整个文档中定位的块：
<chunk>
{chunk_text}
</chunk>

任务：为这个块生成一个上下文描述（50-80字），说明这段内容在整个文档中的位置和作用。{subject_instruction}

要求：
- 必须用简体中文回答
- 必须说明这段内容位于文档的哪个部分（如：文档开头、第二部分、结尾部分、财务指标部分等）
- 必须包含主体名称（公司名称/法院名称/医院名称）
- 如果是法律文档，必须包含法院名称和被告人/当事人姓名
- 如果是病历文档，必须包含医院名称和患者姓名
- 禁止直接复制原文内容
- 禁止使用代词如"该公司"、"该患者"、"本文档"等

格式示例：「这段内容位于[文档位置]，描述了[主体名称]的[主要内容]。」

请直接输出上下文描述："""
    else:
        subject_instruction = ""
        if subject_name:
            subject_instruction = f'\nIMPORTANT: The subject of this document is "{subject_name}".'
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Task: Generate a short context (30-50 words) describing WHERE this chunk is located in the document and its role.{subject_instruction}

Requirements:
- You MUST respond in English only
- MUST specify the location in the document (e.g., "at the beginning", "in the financial section", "at the end")
- MUST include the explicit subject name (company name/court name/hospital name)
- For legal documents, include both court name and defendant/party names
- For medical records, include both hospital name and patient name
- DO NOT copy the original text directly
- DO NOT use pronouns like "the company", "this document", "it", etc.

Format example: "This section, located in [document position], describes [subject name]'s [main content]."

Output ONLY the context description:"""
    
    try:
        response = client.generate(
            model=ollama_config["model"],  # Uses your configured model
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
            }
        )
        context = response["response"].strip()
        return context
    except Exception as e:
        print(f"Error generating chunk context: {e}")
        return ""



def _recursive_split(text, chunk_size, chunk_overlap, language="en"):
    if language == "zh":
        separators = ["\n\n", "\n", "。", "！", "？", " ", ""]
    else:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", "。", "！", "？", " ", ""]      
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

