"""
Selector module will select relevant prompt templates based on the query and relevant chunks.
"""
import os
from utils import llm_generate

def load_templates(template_dir: str) -> dict:
    """Loads all templates from the given directory."""
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(template_dir, filename), 'r') as f:
                templates[filename] = f.read()
    return templates


def select_prompt(query: str, context_chunks: list) -> str:
    """Load all templates and use LLM to select the most relevant one."""
    templates = {}
    template_dir = "template_pool"
    for filename in os.listdir(template_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(template_dir, filename), 'r') as f:
                templates[filename] = f.read()

    """Selects the most relevant prompt template."""
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    template_options = ""
    for name, content in templates.items():
        template_options += f"Template `{name}`:\n{content}\n\n"

    prompt = f"""
    """
    
    chosen_template_name = llm_generate(prompt).strip()
    return templates.get(chosen_template_name, "Template not found")


# test the function
if __name__ == "__main__":
    query = ""
    context_chunks = [
    ]
    chosen_template = select_prompt(query, context_chunks)
    print("Chosen Template:", chosen_template)