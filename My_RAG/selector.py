"""
Selector module will select relevant prompt templates based on the query and relevant chunks.
"""
import os
from utils import llm_generate

def select_prompt(query: str, context_chunks: list) -> str:
    """Load all templates and use LLM to select the most relevant one."""
    templates = {}
    # Use absolute path based on this file's location
    template_dir = os.path.join(os.path.dirname(__file__), "template_pool")
    for filename in os.listdir(template_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(template_dir, filename), 'r') as f:
                templates[filename] = f.read()

    """Selects the most relevant prompt template."""
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    template_options = ""
    for name, content in templates.items():
        # 為了節省 Token，這裡其實可以只放 Template 的 "用途描述" 而非全文
        # 但為了精準度，我們先放前 200 個字或是全文
        preview = content[:200].replace("\n", " ")
        template_options += f"- Filename: `{name}`\n  Content Preview: {preview}...\n\n"


    prompt = f"""
    You are an expert AI Router. Your goal is to select the best prompt template for the user's request based on the context.

    Here are the available templates and their specific use cases:
    
    1. `qa_expert.txt`: Use this when the user asks a **specific question** looking for a precise answer (e.g., "What is the revenue?", "When was the merger?").
    2. `summary_report.txt`: Use this when the user asks for a **summary**, overview, or general understanding of the company (e.g., "Summarize the report", "What happened in 2017?").
    3. `data_extraction.txt`: Use this when the user asks to **extract data**, create a **table**, list specific metrics, or format information structurally (e.g., "Make a table of financial metrics", "List all acquisitions").

    User Query: "{query}"
    Context Preview: "{context[:300]}..."
    
    Answer strictly with the filename only (e.g., 'qa_expert.txt').
    """
    
    chosen_template_name = llm_generate(prompt).strip()
    
    chosen_template_name = chosen_template_name.replace("'", "").replace('"', "").replace("`", "")
    print(f"[System] Selector chose: {chosen_template_name}")
    return templates.get(chosen_template_name, "Error: Selected template not found in pool.")


# test the function
if __name__ == "__main__":

    # query = "When did Green Fields Agriculture Ltd. appoint a new CEO?"
    # query = "How did the senior management changes in March 2021, including the appointment of a new CEO in January 2021 and the expansion of farmland in February 2021, contribute to the enhancement of Green Fields Agriculture Ltd.'s market competitiveness?"
    # query = "According to the hospitalization records of Parker General Hospital, summarize the present illness of Y. Evans."
    # ❌ query = "绿源环保有限公司在2017年4月发生了什么事故？"
    # ❌ query = "绿源环保有限公司在2017年12月设立了什么子公司？"
    query = "比较美好家政服务有限公司和文化传媒有限公司分别发生道德与诚信事件的时间，哪家公司事件发生时间更早？"

    context_chunks = [
        {'page_content': "The Q3 revenue increased by 20%. Operating costs were reduced by 5%."},
        {'page_content': "New product launch is scheduled for next month."}
    ]
    
    prompt_content = select_prompt(query, context_chunks)
    
    print("\n" + "="*30)
    print("Final Selected Template Content:")
    print(prompt_content)
    print("="*30)