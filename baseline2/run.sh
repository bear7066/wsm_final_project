python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_zh.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language zh --output ./predictions/predictions_zh.jsonl
python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_en.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language en --output ./predictions/predictions_en.jsonl

python3 rageval/evaluation/main.py --input_file ./predictions/predictions_zh.jsonl --output_file ./result/score_zh.jsonl --language zh
python3 rageval/evaluation/main.py --input_file ./predictions/predictions_en.jsonl --output_file ./result/score_en.jsonl --language en

python3 rageval/evaluation/process_intermediate.py