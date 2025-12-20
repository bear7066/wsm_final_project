from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
import json
import argparse
from ollama import Client
from openai import OpenAI
from utils import load_ollama_config, load_jsonl

def _generate_chunk_context(language, doc_text, chunk_text, metadata=None, model_name=None, provider="ollama", api_key=None):
    """
    Generate contextual description for a specific chunk using Ollama or OpenAI.
    This follows Anthropic's Contextual Retrieval approach.
    Adapted to accept a specific model_name and provider.
    """
    
    # Truncate doc_text to avoid token limits
    # gpt-4o-mini has 128k tokens context. 
    # 50,000 Chinese characters is roughly 30k-50k tokens (safe with buffer).
    # This provides plenty of context without risking 'Lost in the Middle' or OOM.
    if provider == "openai":
        max_chars = 50000 
    else:
        max_chars = 6000 
        
    doc_text_truncated = doc_text[:max_chars]
    
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
            subject_instruction = f"\n重要：本文档的主体是「{subject_name}」。"
        
        prompt_content = f"""<document>
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

格式示例：「这段内容位于[文档位置]，描述了[主体名称]的[主要內容]。」

请直接输出上下文描述："""
    else:
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\nIMPORTANT: The subject of this document is \"{subject_name}\"."
        
        prompt_content = f"""<document>
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
        if provider == "openai":
            if not api_key:
                # Try getting from env if not provided
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set.")
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name if model_name else "gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.0,
            )
            context = response.choices[0].message.content.strip()
            return context
            
        else: # ollama
            ollama_config = load_ollama_config()
            model_to_use = model_name if model_name else ollama_config.get("model", "granite-code:3b")
            client = Client(host=ollama_config["host"])
            
            response = client.generate(
                model=model_to_use, 
                prompt=prompt_content,
                stream=False,
                options={
                    "temperature": 0.0,
                    "num_ctx": 8192,
                }
            )
            context = response["response"].strip()
            return context

    except Exception as e:
        print(f"Error generating chunk context ({provider}): {e}")
        return ""

def recursive_chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200, model_name=None, output_path=None, provider="ollama", api_key=None):
    """
    Split documents into chunks using recursive character splitting.
    If output_path is provided, chunks are saved to that file.
    """
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name if model_name else 'Default'}")
    
    if output_path and os.path.exists(output_path):
        print(f"Output file already exists at {output_path}")

    if language == "en":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", "：", " ", ""])
    
    chunks = []
    
    for doc in tqdm(docs, desc="Recursive Chunking"):   
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            original_text = doc['content']
            lang = doc['language']

            if lang == language:
                # Generate contextual summary for the document
                meta = doc.copy()
                meta.pop("content", None)
                
                try:
                    split_texts = text_splitter.split_text(original_text)
                    for i, text_chunk in enumerate(split_texts):
                        if text_chunk.strip():
                            # Save original content
                            chunk_meta = meta.copy()
                            chunk_meta['original_content'] = text_chunk
                            chunk_meta['chunk_index'] = i
                            
                            # Generate context
                            context = _generate_chunk_context(
                                lang, 
                                original_text, 
                                text_chunk, 
                                chunk_meta, 
                                model_name=model_name,
                                provider=provider,
                                api_key=api_key
                            )
                            
                            if context:
                                chunk_meta['contextual_prefix'] = context
                            
                            chunks.append({
                                "page_content": context + "\n" + text_chunk if context else text_chunk,
                                "metadata": chunk_meta,
                            })
                except Exception as e:
                    print(f"Error chunking doc: {e}")
    
    print(f"Created {len(chunks)} chunks")
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks to {output_path}")
        
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Generate contextual chunks with a specific model.")
    parser.add_argument('--docs_path', required=True, help='Path to the documents file (jsonl)')
    parser.add_argument('--language', required=True, choices=['zh', 'en'], help='Language to process')
    parser.add_argument('--model', required=True, help='Model name to use (e.g., gpt-oss-20b, gpt-4o-mini)')
    parser.add_argument('--output', required=True, help='Output path for the json cache file')
    parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=512//5, help='Chunk overlap')
    parser.add_argument('--provider', choices=['ollama', 'openai'], default='ollama', help='LLM provider')
    parser.add_argument('--api_key', help='API key for OpenAI (optional if OPENAI_API_KEY env var is set)')
    
    args = parser.parse_args()
    
    print("Loading documents...")
    docs = load_jsonl(args.docs_path)
    print(f"Loaded {len(docs)} documents.")
    
    recursive_chunk_documents(
        docs, 
        args.language, 
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap, 
        model_name=args.model,
        output_path=args.output,
        provider=args.provider,
        api_key=args.api_key
    )

if __name__ == "__main__":
    main()
