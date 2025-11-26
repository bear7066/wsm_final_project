import jsonlines

def load_jsonl(file_path):
    docs = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            docs.append(obj)
    return docs

def save_jsonl(file_path, data):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)