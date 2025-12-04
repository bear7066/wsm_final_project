# Pyserini Retriever 安裝和使用指南

## 📋 前置需求

### 1. 安裝 Java (JDK 11 或更高版本)

**macOS:**
```bash
# 檢查是否已安裝 Java
java -version

# 如果未安裝，使用 Homebrew 安裝
brew install openjdk@11

# 設定環境變數（添加到 ~/.zshrc 或 ~/.bash_profile）
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
export PATH=$JAVA_HOME/bin:$PATH
```

**驗證安裝:**
```bash
java -version
# 應該顯示 Java 11 或更高版本
```

### 2. 安裝 Pyserini

```bash
pip install pyserini
```

**可選：安裝額外的中文分詞工具**
```bash
pip install jieba  # 如果還沒安裝
```

---

## 🚀 使用方法

### 方法 1: 直接替換 (推薦)

在 `main.py` 中修改 import：

```python
# 原來的寫法
# from retriever import create_retriever

# 改用 Pyserini
from pyserini_retriever import create_retriever
```

其他代碼**不需要修改**，因為 API 是兼容的！

---

### 方法 2: 並行比較

如果想比較兩種 retriever 的效果：

```python
from retriever import create_retriever as create_bm25_retriever
from pyserini_retriever import create_retriever as create_pyserini_retriever

# 創建兩個 retriever
bm25_retriever = create_bm25_retriever(chunks, language)
pyserini_retriever = create_pyserini_retriever(chunks, language)

# 比較結果
bm25_results = bm25_retriever.retrieve(query_text)
pyserini_results = pyserini_retriever.retrieve(query_text)
```

---

### 方法 3: 使用持久化索引 (生產環境推薦)

```python
from pyserini_retriever import create_retriever

# 第一次運行：建立並保存索引
retriever = create_retriever(
    chunks, 
    language,
    index_dir="./pyserini_index",  # 指定索引保存路徑
    keep_index=True                 # 保持索引
)

# 後續運行：直接載入已有的索引（快很多！）
# 只要 chunks 沒改變，索引會自動重用
retriever = create_retriever(
    chunks, 
    language,
    index_dir="./pyserini_index",
    keep_index=True
)
```

---

## 🔧 高級功能

### 1. 獲取檢索分數

```python
from pyserini_retriever import PyseriniRetriever

retriever = PyseriniRetriever(chunks, language)

# 獲取結果和 BM25 分數
results_with_scores = retriever.retrieve_with_scores(query_text, top_k=5)

for chunk, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Content: {chunk['page_content'][:100]}...")
    print("-" * 50)
```

### 2. 調整 BM25 參數

在 `pyserini_retriever.py` 中修改：

```python
# 第 55 行附近
self.searcher.set_bm25(k1=0.9, b=0.4)  # 默認值

# 調整建議：
# - k1 控制詞頻飽和度 (通常 0.6-2.0)
# - b 控制文檔長度正規化 (通常 0.3-0.9)
# 中文可能需要更低的 b 值，例如：
self.searcher.set_bm25(k1=1.2, b=0.3)
```

---

## 📊 性能比較

| 特性 | rank_bm25 | Pyserini |
|------|-----------|----------|
| 速度 (小數據集 <10k docs) | 快 | 中等 |
| 速度 (大數據集 >100k docs) | 慢 | **很快** |
| 內存使用 | 高 | 低 |
| 索引持久化 | ❌ | ✅ |
| 中文支援 | 手動 jieba | 內建 CJK |
| 標準化 | - | 學術標準 |
| 安裝複雜度 | 簡單 | 需要 Java |

---

## 🐛 常見問題

### Q1: 報錯 "Java not found"
**A:** 確保安裝了 Java 11+ 並設定了 JAVA_HOME

### Q2: 索引建立很慢
**A:** 
- 第一次建立索引會比較慢，之後會重用
- 使用 `index_dir` 參數保存索引
- 確保 `keep_index=True`

### Q3: 中文檢索效果不好
**A:**
- 確保設定了正確的語言：`language='zh'`
- 嘗試調整 BM25 參數（降低 b 值）
- 確保文檔在切分時保留了完整的語義

### Q4: 記憶體不足
**A:**
- Pyserini 使用磁盤索引，記憶體使用應該比 rank_bm25 少
- 如果還是不夠，考慮減少 `storeDocvectors` 選項

---

## 📝 完整範例

```python
from pyserini_retriever import create_retriever
from utils import load_jsonl
from chunker import chunk_documents

# 載入數據
docs = load_jsonl("documents.jsonl")
queries = load_jsonl("queries.jsonl")

# 切分文檔
chunks = chunk_documents(docs, language="zh")

# 創建 retriever（第一次會建立索引）
retriever = create_retriever(
    chunks, 
    language="zh",
    index_dir="./index",
    keep_index=True
)

# 檢索
for query in queries:
    query_text = query['query']['content']
    results = retriever.retrieve(query_text, top_k=5)
    
    print(f"\nQuery: {query_text}")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk['page_content'][:100]}...")
```

---

## 🎯 建議

1. **開發階段**: 使用 `rank_bm25`（快速迭代）
2. **性能測試**: 切換到 `pyserini_retriever`（比較效果）
3. **生產部署**: 使用 `pyserini_retriever` + 持久化索引

---

## 📚 更多資源

- [Pyserini 官方文檔](https://github.com/castorini/pyserini)
- [BM25 參數調優指南](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)
