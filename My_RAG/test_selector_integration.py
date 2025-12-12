from selector import select_prompt
from generator import generate_answer

def test_integration():
    # Test cases
    queries = [
        "What was the revenue of Company A in 2021?", # Should be data_extraction
        "Compare the performance of Model X and Model Y.", # Should be comparison
        "Summarize the key risks mentioned in the report.", # Should be summary_report
        "How does the algorithm work?" # Should be qa_expert
    ]
    
    context_chunks = [{"page_content": "Company A had a revenue of $10 million in 2021. Model X is faster than Model Y. The report mentions market volatility as a key risk."}]

    print("=== Testing Selector & Generator Integration ===")
    
    for q in queries:
        print(f"\nQuery: {q}")
        
        # 1. Select Template
        template_content = select_prompt(q)
        print(f"Template Selected: {'Yes' if template_content else 'No'}")
        if template_content:
            print(f"Template Preview: {template_content[:50].replace(chr(10), ' ')}...")
        
        # 2. Generate Answer
        try:
            answer = generate_answer(q, context_chunks, template_content)
            print("Generation: Success")
            # print(f"Answer: {answer}") 
        except Exception as e:
            print(f"Generation Failed: {e}")

if __name__ == "__main__":
    test_integration()
