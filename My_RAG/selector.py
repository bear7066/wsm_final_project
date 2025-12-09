"""
Selector module will select relevant prompt templates based on the query and relevant chunks.
"""
import os
#from utils import llm_generate
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Classifier:
    def __init__(self, model_path):
        """
        初始化分類器
        param model_path: my_query_classifier/my_domain_classifier
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"fail to load model")

        self.id2label = self._load_label_mapping(model_path)
        #print(f"{len(self.id2label)} classes")

    def _load_label_mapping(self, model_path):
        raw_id2label = {}
        pkl_path = os.path.join(model_path, "label_mapping.pkl")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    raw_id2label = pickle.load(f)
            except Exception as e:
                print(f"fail to load label_mapping.pkl")

        if not raw_id2label:
            raw_id2label = self.model.config.id2label

        clean_id2label = {}
        for k, v in raw_id2label.items():
            try:
                clean_id2label[int(k)] = v
            except ValueError:
                continue

        return clean_id2label

    def predict(self, text):
        """
        param text: Query Content
        return: query type/domain (String)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()

        if predicted_class_id in self.id2label:
            return self.id2label[predicted_class_id]
        else:
            return f"Unknown_Label_{predicted_class_id}"

class GetPrompt:
    def __init__(self, query, domain, query_type, relevant_docs):
        self.domain = domain
        self.query = query
        self.query_type = query_type
        self.relevant_docs = relevant_docs
        self.context_instruction = """
INSTRUCTIONS FOR ANSWERING:
1. The answer must use the following format:
***Your answer***
[References]
***The parts of context where you get your answer from*** 
Example:
InnovateTech, Inc. reduced more liabilities through debt restructuring with a reduction of $30 million compared to Culture Innovators Ltd.'s $15 million.
[References]
"In July 2021, Culture Innovators Ltd. underwent a successful debt restructuring, resulting in a reduction of liabilities by $15 million.", "The company's total liabilities were reduced to $80 million following the successful debt restructuring, improving its financial condition."
2. The following text contains multiple snippets from retrieved documents, wrapped in <document index="X"> tags.
3. You must answer the user's query using ONLY the information provided in these documents.
4. If different documents contain conflicting information, prioritize the one that seems more recent or authoritative, and mention the conflict.
5. Do not include the XML tags in your final answer.
        """
        self.ch_context_instruction = """
答题说明：
1. 答案必须采用以下格式：
***您的答案***
[References]
***作为答案依据的一个或多个内文片段***
例如：
刘某和张某。
[References]
"追尾事故导致前方车辆上的两名乘客刘某和张某轻微受伤，车辆也不同程度地损毁。"
2. 以下文本包含多个从检索到的文档中摘录的片段，这些片段包含在 <document index="X"> 标签中。
3. 您必须仅使用这些文档中提供的信息来回答用户的问题。
4. 如果不同文档包含相互矛盾的信息，请优先采用看起来更新或更权威的文档，并说明矛盾之处。
5. 最终答案中请勿包含 XML 标签。
        """
        self.TYPE_PROMPTS = {
            "Factual Question": "type5.txt",
            "Multi-hop Reasoning Question": "type1.txt",
            "Summary Question": "type4.txt",
            "Irrelevant Unsolvable Question": "type6.txt",
            "Multi-document Information Integration Question": "type3.txt",
            "Multi-document Comparison Question": "type2.txt",
            "Multi-document Time Sequence Question": "type7.txt",
            "Summarization Question": "type4.txt",
            "无关无解问": "ch_type6.txt",
            "事实性问题": "ch_type5.txt",
            "多跳推理问题": "ch_type1.txt",
            "总结性问题": "ch_type4.txt",
            "多文档信息整合问题": "ch_type3.txt",
            "多文档时间序列问题": "ch_type7.txt",
            "多文档对比问题": "ch_type2.txt"
        }
        self.EN_DOMAIN_PROMPTS = {"Finance": "fin.txt", "Law": "law.txt", "Medical": "med.txt"}
        self.CH_DOMAIN_PROMPTS = {"Finance": "ch_fin.txt", "Law": "ch_law.txt", "Medical": "ch_med.txt"}

    def format_docs(self):
        """
        add XML Tags
        要再依retrieve return格式修改
        是否需要包含chunk所屬文件的doc id
        """
        formatted_string = "<documents>\n"

        for i, chunk in enumerate(self.relevant_docs):
            doc_index = i + 1
            content = chunk.get('content', '').strip()
            source = chunk.get('metadata', {}).get('source', 'unknown')

            formatted_string += f"""
        <document index="{doc_index}">
            <source>{source}</source>
            <content>
    {content}
            </content>
        </document>\n"""

        formatted_string += "</documents>"
        return formatted_string

    def output(self):
        file_name = self.TYPE_PROMPTS.get(self.query_type)
        f = open(os.path.join('./template_pool/', file_name), "r", encoding="utf8")
        type_instruction = f.read()
        f.close()
        #formatted_context = self.format_docs()
        formatted_context = "test"
        if self.query_type == "Factual Question" or self.query_type == "Multi-hop Reasoning Question" or self.query_type == "Summary Question" or self.query_type == "Irrelevant Unsolvable Question" or self.query_type == "Multi-document Information Integration Question" or self.query_type == "Multi-document Comparison Question" or self.query_type == "Multi-document Time Sequence Question" or self.query_type == "Summarization Question":
            file_name = self.EN_DOMAIN_PROMPTS.get(self.domain)
            f = open(os.path.join('./template_pool/', file_name), "r", encoding="utf8")
            domain_instruction = f.read()
            f.close()
            final_prompt = f"""\r{self.context_instruction}\r{domain_instruction}\rTASK: {type_instruction}
CONTEXT:
{formatted_context}
PROBLEM:
{self.query}
Please provide your response following the ROLE and STYLE defined above.
"""
        else:
            file_name = self.CH_DOMAIN_PROMPTS.get(self.domain)
            f = open(os.path.join('./template_pool/', file_name), "r", encoding="utf8")
            domain_instruction = f.read()
            f.close()
            final_prompt = f"""\r{self.ch_context_instruction}\r{domain_instruction}\r任务: {type_instruction}
內文:
{formatted_context}
问题:
{self.query}
请按照上述角色和风格提供您的回复。
"""
        return final_prompt


if __name__ == "__main__":
    q_classifier = Classifier(model_path="./my_query_classifier")
    d_classifier = Classifier(model_path="./my_domain_classifier")

    q1 = "According to the judgment of Northwood, Richmond, Court, what is the favorite color of the defendant S. Taylor?"
    print(f"Q: {q1} -> Type: {q_classifier.predict(q1)}")
    print(f"Q: {q1} -> Domain: {d_classifier.predict(q1)}")
    gp1 = GetPrompt(q1, d_classifier.predict(q1), q_classifier.predict(q1), "test docs")
    print(gp1.output())
    q2 = "根据彩虹市桐城区人民法院的判决书，总结该案的法律程序及审判结果。"
    print(q_classifier.predict(q2))
    print(d_classifier.predict(q2))
    gp2 = GetPrompt(q2, d_classifier.predict(q2), q_classifier.predict(q2), "test docs")
    print(gp2.output())