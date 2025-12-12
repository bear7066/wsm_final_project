
import json
import jsonlines

def validate_jsonl(file_path):
    print(f"Validating {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error on line {i+1}: {e}")
                    print(f"Line content (repr): {repr(line)}")
                    return
        print("Validation complete. No errors found with json.loads.")
    except Exception as e:
        print(f"File validation failed: {e}")

if __name__ == "__main__":
    validate_jsonl("predictions/predictions_zh.jsonl")
