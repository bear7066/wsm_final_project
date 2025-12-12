import os
from utils import llm_generate

def load_template(template_name):
    # Construct absolute path to template_pool
    # Assuming selector.py is in My_RAG/, and template_pool is in My_RAG/template_pool/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "template_pool", f"{template_name}.txt")
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Template {template_name} not found at {template_path}. Using basic prompt.")
        return None

def select_prompt(query):
    # 1. Classify the query
    classification_prompt = f"""You are a query classifier. Please analyze the user's question and categorize it into ONE of the following types:

    1. "data_extraction": The user asks to extract specific data points (e.g., "What was the revenue in 2021?", "Extract the CEO's name").
    2. "comparison": The user asks to compare two or more things (e.g., "Compare the revenue of Company A and B", "Difference between 2020 and 2021").
    3. "summary_report": The user asks for a summary or overview of a topic/document (e.g., "Summarize the risks", "Give an overview of the report").
    4. "qa_expert": General question answering that requires synthesis or detailed explanation, or doesn't fit the above.

    User Question: {query}

    Output ONE word only: data_extraction, comparison, summary_report, or qa_expert.
    """
    
    category = llm_generate(classification_prompt).strip().lower()
    
    # Clean up response just in case (remove punctuation etc)
    category = category.strip('"').strip("'").strip(".")
    
    valid_categories = ["data_extraction", "comparison", "summary_report", "qa_expert"]
    if category not in valid_categories:
        print(f"Warning: LLM returned invalid category '{category}'. Defaulting to 'qa_expert'.")
        category = "qa_expert"
        
    print(f"Query classified as: {category}")
    
    # 2. Load the corresponding template
    template_content = load_template(category)
    
    # Fallback if template loading fails
    if not template_content:
        # Return a default simple template or None to let generator use its default
        return None
        
    return template_content
