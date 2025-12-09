# Hybrid Retrieval Strategy 實現說明

## 📋 概述

這個實現結合了兩種檢索策略：
1. **BM25 (Sparse Retrieval)** - 擅長精確關鍵字匹配
2. **Vector Embeddings (Dense Retrieval)** - 擅長語義理解

## 🏗️ 架構設計

### 核心思路
```
查詢 → BM25 Retriever → Top-K 候選 + 分數
     ↓
     → Vector Retriever → Top-K 候選 + 分數
     ↓
     → 分數歸一化 (0-1)
     ↓
     → 加權合併 (Weighted Sum)
     ↓
     → 最終排序結果
```

### 實現細節

#### 1. 兩個獨立的 Retriever

**BM25Retriever** (`retriever.py`)
- 使用 `rank_bm25` 庫
- 中文使用 jieba 分詞
- 分數範圍：0 到無限大

**VectorRetriever** (`hybrid_retriever.py`)
- 使用 `sentence-transformers`
- 中文模型：`BAAI/bge-small-zh-v1.5`
- 英文模型：`BAAI/bge-small-en-v1.5`
- 分數範圍：0 到 1（cosine similarity）

#### 2. 分數歸一化

使用 **Min-Max Normalization** 將不同範圍的分數統一到 0-1：

```python
normalized_score = (score - min_score) / (max_score - min_score)
```

#### 3. 加權合併

```python
final_score = bm25_weight * normalized_bm25_score + vector_weight * normalized_vector_score
```

## 🎯 使用方法

### 基本用法

```python
from hybrid_retriever import create_retriever

# 創建 hybrid retriever
retriever = create_retriever(
    chunks,
    language='zh',
    bm25_weight=0.5,      # BM25 權重
    vector_weight=0.5     # Vector 權重
)

# 檢索
results = retriever.retrieve(query, top_k=5)
```

### 權重配置建議

| 場景 | BM25 權重 | Vector 權重 | 說明 |
|------|-----------|-------------|------|
| **平衡** | 0.5 | 0.5 | 適合大多數情況 |
| **關鍵字重** | 0.7 | 0.3 | 查詢包含專有名詞、數字、代碼 |
| **語義重** | 0.3 | 0.7 | 查詢是自然語言問題 |

### 在 main.py 中使用

```python
from hybrid_retriever import create_retriever

# 創建 retriever（已在 main.py 中配置）
retriever = create_retriever(chunks, language)

# 檢索
candidate_chunks = retriever.retrieve(query_text, top_k=10)
```

## 📊 性能對比

根據測試結果：

| 策略 | 初始化時間 | 檢索速度 | 精確匹配 | 語義理解 |
|------|-----------|---------|---------|---------|
| BM25 Only | 快 | 快 | ⭐⭐⭐ | ⭐ |
| Vector Only | 慢 | 中 | ⭐ | ⭐⭐⭐ |
| **Hybrid** | 慢 | 中 | ⭐⭐⭐ | ⭐⭐⭐ |

## 🔧 調優建議

### 1. 調整權重

根據您的數據特性調整權重：

```python
# 測試不同配置
configs = [
    (0.5, 0.5),  # 平衡
    (0.6, 0.4),  # 偏向 BM25
    (0.4, 0.6),  # 偏向 Vector
]

for bm25_w, vector_w in configs:
    retriever = create_retriever(chunks, 'zh', bm25_w, vector_w)
    # 評估性能...
```

### 2. 選擇不同的 Embedding 模型

```python
# 使用其他模型
retriever = create_retriever(
    chunks,
    language='zh',
    embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)
```

### 3. 調整候選數量

```python
# 從更多候選中選擇
results = retriever.retrieve(query, top_k=5, candidate_k=20)
```

## 📝 完整工作流程

```python
# 1. 載入文檔
docs = load_jsonl(docs_path)

# 2. 分塊
chunks = chunk_documents(docs, language)

# 3. 創建 hybrid retriever
retriever = create_retriever(
    chunks, 
    language,
    bm25_weight=0.5,
    vector_weight=0.5
)

# 4. 檢索
for query in queries:
    # 獲取候選
    candidates = retriever.retrieve(query, top_k=10)

    # (可選) LLM 重排序
    # final_results = rerank_chunks(query, candidates, language, top_k=5)

    # 生成答案
    answer = generate_answer(query, candidates, prompt_template, language)
```

## 🚀 運行評估

```bash
# 使用 hybrid retriever 運行完整評估
./run_evaluate.sh
```

## 📦 依賴項

確保已安裝：
```bash
pip install sentence-transformers torch rank_bm25 jieba
```

## 💡 優勢

1. **互補性**：BM25 和 Vector 的優勢互補
2. **靈活性**：可調整權重適應不同場景
3. **簡單性**：兩個獨立 retriever，易於理解和維護
4. **可擴展性**：可以輕鬆添加更多檢索策略

## ⚠️ 注意事項

1. **初始化時間**：Vector retriever 需要時間建立 embeddings
2. **記憶體使用**：embeddings 會佔用記憶體
3. **模型下載**：首次使用會下載 embedding 模型（~100MB）

## 🎓 進階技巧

### 動態權重調整

根據查詢類型動態調整權重：

```python
def get_weights(query):
    # 如果查詢包含數字或專有名詞，增加 BM25 權重
    if any(char.isdigit() for char in query):
        return 0.7, 0.3
    # 否則使用平衡權重
    return 0.5, 0.5

bm25_w, vector_w = get_weights(query)
retriever = create_retriever(chunks, language, bm25_w, vector_w)
```

### 結合 LLM 重排序

```python
# 1. Hybrid 檢索獲取候選
candidates = retriever.retrieve(query, top_k=20)

# 2. LLM 重排序精選 top-5
final_results = rerank_chunks(query, candidates, language, top_k=5)
```