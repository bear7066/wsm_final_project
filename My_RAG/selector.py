"""
Selector module will select relevant prompt templates based on the query and relevant chunks.
"""
import os
#from utils import llm_generate
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from generator import generate_answer
from utils import llm_generate

# class Classifier:
#     def __init__(self, model_path):
#         """
#         初始化分類器
#         param model_path: my_query_classifier/my_domain_classifier
#         """
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#             self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
#             self.model.eval()
#         except Exception as e:
#             raise ValueError(f"fail to load model")

#         self.id2label = self._load_label_mapping(model_path)
#         #print(f"{len(self.id2label)} classes")

#     def _load_label_mapping(self, model_path):
#         raw_id2label = {}
#         pkl_path = os.path.join(model_path, "label_mapping.pkl")

#         if os.path.exists(pkl_path):
#             try:
#                 with open(pkl_path, "rb") as f:
#                     raw_id2label = pickle.load(f)
#             except Exception as e:
#                 print(f"fail to load label_mapping.pkl")

#         if not raw_id2label:
#             raw_id2label = self.model.config.id2label

#         clean_id2label = {}
#         for k, v in raw_id2label.items():
#             try:
#                 clean_id2label[int(k)] = v
#             except ValueError:
#                 continue

#         return clean_id2label

#     def predict(self, text):
#         """
#         param text: Query Content
#         return: query type/domain (String)
#         """
#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=128,
#             padding=True
#         ).to(self.device)

#         with torch.no_grad():
#             logits = self.model(**inputs).logits

#         predicted_class_id = logits.argmax().item()

#         if predicted_class_id in self.id2label:
#             return self.id2label[predicted_class_id]
#         else:
#             return f"Unknown_Label_{predicted_class_id}"

# class GetPrompt:
#     def __init__(self, query, domain, query_type, relevant_docs, language='en'):
#         self.domain = domain
#         self.query = query
#         self.query_type = query_type
#         self.relevant_docs = relevant_docs
#         self.language = language
#         self.context_instruction = """
# INSTRUCTIONS FOR ANSWERING:
# 1. The answer must use the following format:
# ***Your answer***
# [References]
# ***The parts of context where you get your answer from*** 
# Example:
# InnovateTech, Inc. reduced more liabilities through debt restructuring with a reduction of $30 million compared to Culture Innovators Ltd.'s $15 million.
# [References]
# "In July 2021, Culture Innovators Ltd. underwent a successful debt restructuring, resulting in a reduction of liabilities by $15 million.", "The company's total liabilities were reduced to $80 million following the successful debt restructuring, improving its financial condition."
# 2. The following text contains multiple snippets from retrieved documents, wrapped in <document index="X"> tags.
# 3. You must answer the user's query using ONLY the information provided in these documents.
# 4. If different documents contain conflicting information, prioritize the one that seems more recent or authoritative, and mention the conflict.
# 5. Do not include the XML tags in your final answer.
#         """
#         self.ch_context_instruction = """
# 答题说明：
# 1. 答案必须采用以下格式：
# ***您的答案***
# [References]
# ***作为答案依据的一个或多个内文片段***
# 例如：
# 刘某和张某。
# [References]
# "追尾事故导致前方车辆上的两名乘客刘某和张某轻微受伤，车辆也不同程度地损毁。"
# 2. 以下文本包含多个从检索到的文档中摘录的片段，这些片段包含在 <document index="X"> 标签中。
# 3. 您必须仅使用这些文档中提供的信息来回答用户的问题。
# 4. 如果不同文档包含相互矛盾的信息，请优先采用看起来更新或更权威的文档，并说明矛盾之处。
# 5. 最终答案中请勿包含 XML 标签。
#         """
#         self.TYPE_PROMPTS = {
#             "Factual Question": "type5.txt",
#             "Multi-hop Reasoning Question": "type1.txt",
#             "Summary Question": "type4.txt",
#             "Irrelevant Unsolvable Question": "type6.txt",
#             "Multi-document Information Integration Question": "type3.txt",
#             "Multi-document Comparison Question": "type2.txt",
#             "Multi-document Time Sequence Question": "type7.txt",
#             "Summarization Question": "type4.txt",
#             "无关无解问": "ch_type6.txt",
#             "事实性问题": "ch_type5.txt",
#             "多跳推理问题": "ch_type1.txt",
#             "总结性问题": "ch_type4.txt",
#             "多文档信息整合问题": "ch_type3.txt",
#             "多文档时间序列问题": "ch_type7.txt",
#             "多文档对比问题": "ch_type2.txt"
#         }
#         self.EN_DOMAIN_PROMPTS = {"Finance": "fin.txt", "Law": "law.txt", "Medical": "med.txt"}
#         self.CH_DOMAIN_PROMPTS = {"Finance": "ch_fin.txt", "Law": "ch_law.txt", "Medical": "ch_med.txt"}

#     def format_docs(self):
#         """
#         add XML Tags
#         要再依retrieve return格式修改
#         是否需要包含chunk所屬文件的doc id
#         """
#         formatted_string = "<documents>\n"

#         for i, chunk in enumerate(self.relevant_docs):
#             doc_index = i + 1
#             content = chunk.get('page_content', chunk.get('content', '')).strip()
#             source = chunk.get('metadata', {}).get('source', 'unknown')

#             formatted_string += f"""
#         <document index="{doc_index}">
#             <source>{source}</source>
#             <content>
#     {content}
#             </content>
#         </document>\n"""

#         formatted_string += "</documents>"
#         return formatted_string

#     def output(self):
#         file_name = self.TYPE_PROMPTS.get(self.query_type)
#         if file_name is None:
#              # Fallback default if type not found
#              file_name = "type5.txt" if self.language != 'zh' else "ch_type5.txt"

#         # Force correct language template if mismatch
#         if self.language == 'zh':
#              if not file_name.startswith('ch_'):
#                   file_name = 'ch_' + file_name
#         else: # language == 'en' ('zh' is the only special case here)
#              if file_name.startswith('ch_'):
#                   file_name = file_name[3:]

#         # Handle file not found fallback? For now assume files exist.
#         try:
#              f = open(os.path.join('My_RAG/template_pool/', file_name), "r", encoding="utf8")
#              type_instruction = f.read()
#              f.close()
#         except FileNotFoundError:
#              # Fallback to default of that language
#              default_file = "ch_type5.txt" if self.language == 'zh' else "type5.txt"
#              with open(os.path.join('My_RAG/template_pool/', default_file), "r", encoding="utf8") as f:
#                  type_instruction = f.read()

#         formatted_context = self.format_docs()
        
#         if self.language != 'zh':
#             file_name_domain = self.EN_DOMAIN_PROMPTS.get(self.domain, "fin.txt")
#             with open(os.path.join('My_RAG/template_pool/', file_name_domain), "r", encoding="utf8") as f:
#                 domain_instruction = f.read()
            
#             final_prompt = f"""\r{self.context_instruction}\r{domain_instruction}\rTASK: {type_instruction}
# CONTEXT:
# {formatted_context}
# PROBLEM:
# {self.query}
# Please provide your response following the ROLE and STYLE defined above.
# """
#         else:
#             file_name_domain = self.CH_DOMAIN_PROMPTS.get(self.domain, "ch_fin.txt")
#             with open(os.path.join('My_RAG/template_pool/', file_name_domain), "r", encoding="utf8") as f:
#                 domain_instruction = f.read()

#             final_prompt = f"""\r{self.ch_context_instruction}\r{domain_instruction}\r任务: {type_instruction}
# 內文:
# {formatted_context}
# 问题:
# {self.query}
# 请按照上述角色和风格提供您的回复。
# """
#         return final_prompt


def select_prompt(query, language="en"):
    """
    Selects the most appropriate prompt template from a fixed set of 4 options using LLM.
    Options: qa_expert.txt, data_extraction.txt, comparison.txt, summary_report.txt
    """
    
    if language == 'zh':
        prompt_config = f"""
你是一個精準的任務分類助手。請分析使用者的查詢，並從以下四個選項中選擇最適合的提示詞模板檔案名稱：

1. `qa_expert.txt` - 用於一般問答、解釋概念、知識查詢或事實性問題。
2. `data_extraction.txt` - 用於從文本中提取特定數據、數字、實體或結構化信息。
3. `comparison.txt` - 用於比較多個對象、時間點或事件之間的異同。
4. `summary_report.txt` - 用於總結內容、生成摘要或撰寫報告。

使用者查詢: {query}

請輸出 JSON 格式，包含一個鍵 "filename"，值為上述四個檔案名稱之一。不要輸出其他解釋。
例如: {{"filename": "qa_expert.txt"}}
"""
    else:
        prompt_config = f"""
You are a precise task classification assistant. Please analyze the user's query and select the most appropriate prompt template filename from the four options below:

1. `qa_expert.txt` - For general Q&A, explaining concepts, knowledge queries, or factual questions.
2. `data_extraction.txt` - For extracting specific data, numbers, entities, or structured information from text.
3. `comparison.txt` - For comparing similarities and differences between multiple objects, time points, or events.
4. `summary_report.txt` - For summarizing content, generating abstracts, or writing reports.

User Query: {query}

Please output in JSON format with a single key "filename", and the value being one of the four filenames above. Do not output any other explanation.
Example: {{"filename": "qa_expert.txt"}}
"""
    response = llm_generate(prompt_config)
    
    # Simple parsing logic
    selected_file = "qa_expert.txt" # Default
    try:
        import json
        # Try to find JSON object in response
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            data = json.loads(json_str)
            if "filename" in data:
                selected_file = data["filename"]
    except Exception as e:
        print(f"Error parsing prompt selection: {e}. Using default.")
        
    # Validate selection
    valid_options = ["qa_expert.txt", "data_extraction.txt", "comparison.txt", "summary_report.txt"]
    if selected_file not in valid_options:
        # Fallback heuristic
        if "compare" in query.lower() or "difference" in query.lower() or "vs" in query.lower() or "比" in query.lower():
            selected_file = "comparison.txt"
        elif "summar" in query.lower() or "overview" in query.lower() or "總結" in query.lower() or "大綱" in query.lower():
            selected_file = "summary_report.txt"
        elif "extract" in query.lower() or "list" in query.lower() or "data" in query.lower() or "提取" in query.lower() or "列出" in query.lower():
            selected_file = "data_extraction.txt"
        else:
            if language == 'zh':
                selected_file = "ch_qa_expert.txt"
            else:
                selected_file = "qa_expert.txt"
    # Read the content of the selected file
    try:
        import os
        # Adjust filename for Chinese if needed
        if language == 'zh':
            # Check if a ch_ prefixed file exists
            ch_filename = "ch_" + selected_file
            ch_path = os.path.join('My_RAG/template_pool', ch_filename)
            if os.path.exists(ch_path):
                selected_file = ch_filename
                
        print(f"\nSelected template: {selected_file}\n")
        template_path = os.path.join('My_RAG/template_pool', selected_file)
        if not os.path.exists(template_path):
             pass
             
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading template {selected_file}: {e}")
        return "Task: Answer the query based on the context.\nContext:\n{context}\nQuery:\n{query}" # Fallback generic prompt

    return ""


# def initialize_classifiers():
#     q_classifier = Classifier(model_path="My_RAG/my_query_classifier")
#     d_classifier = Classifier(model_path="My_RAG/my_domain_classifier")
#     return q_classifier, d_classifier

# if __name__ == "__main__":
#     # Mock data for testing
#     mock_docs = [{"page_content": "This is a test document snippet.", "metadata": {"source": "test_source"}}]

#     print("=== 1. Testing Manual Prompt Selection Logic (Bypassing Classifiers) ===")
#     # Define test cases: (Query Type, Domain, Desired Language Context)
#     test_cases = [
#         # English Cases
#         ("Factual Question", "Finance", "en"),
#         ("Multi-hop Reasoning Question", "Law", "en"),
#         ("Summary Question", "Medical", "en"),
#         # Chinese Cases
#         ("事实性问题", "Finance", "zh"),
#         ("无关无解问", "Medical", "zh"),
#         ("多文档对比问题", "Law", "zh"),
#     ]

#     for q_type, domain, lang in test_cases:
#         print(f"\n[Test Case] Type: {q_type} | Domain: {domain} | Lang: {lang}")
#         try:
#             gp = GetPrompt("Test Query", domain, q_type, mock_docs, language=lang)
#             prompt = gp.output()
#             # print(f"Output Preview:\n{prompt[:200]}...\n")
#             print(f"✅ Template loaded successfully. Prompt length: {len(prompt)}")
            
#             # Basic verification of content (checking if instructions are included)
#             if "TASK:" in prompt or "任务:" in prompt:
#                  print("   -> Contains task instruction.")
#             if "<document index=\"1\">" in prompt:
#                  print("   -> Contains formatted context.")
                 
#         except Exception as e:
#             print(f"❌ Error generating prompt: {e}")

#     print("\n=== 2. Testing Classifier Integration (Requires Models) ===")
#     q_classifier = None
#     d_classifier = None
#     try:
#         q_classifier = Classifier(model_path="My_RAG/my_query_classifier")
#         d_classifier = Classifier(model_path="My_RAG/my_domain_classifier")

#         queries = [
#             ("According to the judgment of Northwood, Richmond, Court, what is the favorite color of the defendant S. Taylor?", "en"),
#             ("根据彩虹市桐城区人民法院的判决书，总结该案的法律程序及审判结果。", "zh")
#         ]

#         for q, lang in queries:
#             print(f"\nQuery: {q}")
#             q_type = q_classifier.predict(q)
#             domain = d_classifier.predict(q)
#             print(f" -> Predicted Type: {q_type}")
#             print(f" -> Predicted Domain: {domain}")
            
#             gp = GetPrompt(q, domain, q_type, mock_docs, language=lang)
#             prompt = gp.output()
#             print(f" -> Prompt generated successfully (Length: {len(prompt)})")

#     except Exception as e:
#         print(f"\n⚠️ Could not run classifier integration tests (Models might be missing or GPU unavailable).")
#         print(f"Error details: {e}")

#     print("\n=== 3. Testing with Dragonball Dataset Queries (test_queries_en.jsonl) ===")
#     import json
    
#     dataset_path = "dragonball_dataset/test_queries_en.jsonl"
#     if os.path.exists(dataset_path):
#         try:
#             with open(dataset_path, "r", encoding="utf-8") as f:
#                 # Test first 5 queries
#                 for i, line in enumerate(f):
#                     if i >= 5: break
                    
#                     data = json.loads(line)
#                     query_content = data['query']['content']
#                     actual_type = data['query']['query_type']
#                     actual_domain = data['domain']
#                     actual_lang = data.get('language', 'zh') # Default to zh if missing, but dataset has it
                    
#                     print(f"\n[Dragonball Q{i}] Content: {query_content}")
#                     print(f" -> Actual Type: {actual_type} | Actual Domain: {actual_domain} | Lang: {actual_lang}")
                    
#                     try:
#                         # Use classifiers if available
#                         pred_type = q_classifier.predict(query_content)
#                         pred_domain = d_classifier.predict(query_content)
#                         print(f" -> Predicted Type: {pred_type} | Predicted Domain: {pred_domain}")
                        
#                         param_type = pred_type if "Unknown" not in pred_type else actual_type
#                         param_domain = pred_domain if "Unknown" not in pred_domain else actual_domain
                        
#                         gp = GetPrompt(query_content, param_domain, param_type, mock_docs, language=actual_lang)
#                         prompt = gp.output()
#                         print(f" -> Prompt generated (Length: {len(prompt)})")
#                         # print(f" -> Prompt Preview: {prompt[:100].replace(chr(10), ' ')}...")
                        
#                         print(" -> Generating answer...")
#                         answer = generate_answer(query_content, mock_docs, prompt, actual_lang)
#                         print(f" -> Answer: {answer}")
                        
#                     except Exception as e:
#                         print(f" -> ❌ Prediction/Generation failed: {e}")
                        
#         except Exception as e:
#             print(f"Error reading dataset: {e}")
#     else:
#         print(f"Dataset not found at {dataset_path}")

#     print("\n=== 4. Testing select_prompt logic ===")
#     test_queries = [
#         ("What is the revenue difference between 2020 and 2021?", "en"),
#         ("Summarize the key findings of the report.", "en"),
#         ("Extract all financial figures from the document.", "en"),
#         ("How does the system work?", "en"),
#         ("比較兩家公司的營收差異。", "zh")
#     ]
    
#     for q, lang in test_queries:
#         print(f"Query: {q} ({lang})")
#         selected = select_prompt(q, lang)
#         print(f" -> Selected: {selected}")