# 🚀 快速開始：使用 Pyserini Retriever

## ✅ 系統已就緒
- ✓ Java 23.0.1 已安裝
- ✓ pyserini_retriever.py 已創建
- ✓ 原有的 retriever.py 已保留

---

## 📦 步驟 1: 安裝 Pyserini

```bash
cd /Users/leonko/Documents/GitHub/wsm_final_project/My_RAG
pip install pyserini
```

---

## 🔄 步驟 2: 修改 main.py (三選一)

### 選項 A: 完全切換到 Pyserini (推薦)

修改第 4 行：
```python
# from retriever import create_retriever
from pyserini_retriever import create_retriever
```

其他代碼不需要改動！

### 選項 B: 使用持久化索引 (生產環境)

修改第 24-26 行：
```python
# 創建 retriever 並保存索引
print("Creating retriever...")
retriever = create_retriever(
    chunks, 
    language,
    index_dir="./pyserini_index",  # 索引會保存在這裡
    keep_index=True                 # 下次運行會重用
)
print("Retriever created successfully.")
```

### 選項 C: 比較兩種方法

```python
from retriever import create_retriever as create_bm25
from pyserini_retriever import create_retriever as create_pyserini

# 可以切換使用
retriever = create_pyserini(chunks, language)  # 或 create_bm25(chunks, language)
```

---

## 🧪 步驟 3: 測試比較 (可選)

運行比較腳本看看兩種 retriever 的差異：

```bash
python compare_retrievers.py \
    --docs_path 你的文檔路徑.jsonl \
    --query_path 你的查詢路徑.jsonl \
    --language zh
```

這會顯示：
- 初始化時間對比
- 檢索速度對比  
- 結果重疊度
- 推薦使用哪種方法

---

## 📝 步驟 4: 運行你的 RAG 系統

```bash
python main.py \
    --query_path queries.jsonl \
    --docs_path documents.jsonl \
    --language zh \
    --output predictions.jsonl
```

---

## 🎯 關鍵優勢

### Pyserini vs BM25Okapi

| 特性 | 原來的 BM25Okapi | 新的 Pyserini |
|------|-----------------|---------------|
| 小數據集 (< 1k docs) | ⚡ 可能稍快 | 🚀 也很快 |
| 大數據集 (> 10k docs) | 🐌 較慢 | ✨ **快很多** |
| 索引重用 | ❌ 每次重建 | ✅ **可持久化** |
| 記憶體使用 | 📈 高 | 📉 低 |
| 學術標準 | ⚠️ 非標準 | ✅ Lucene 標準 |

---

## 💡 實用技巧

### 1. 第一次運行會慢（建立索引）
```bash
# 第一次運行
python main.py ...  # 可能需要幾分鐘建立索引

# 後續運行（如果使用 index_dir）
python main.py ...  # 秒開！直接載入索引
```

### 2. 調優 BM25 參數 (可選)

在 `pyserini_retriever.py` 第 55 行修改：
```python
# 預設值
self.searcher.set_bm25(k1=0.9, b=0.4)

# 中文文檔可以試試
self.searcher.set_bm25(k1=1.2, b=0.3)
```

### 3. 獲取檢索分數

```python
# 在你的代碼中
results = retriever.retrieve_with_scores(query_text, top_k=5)
for chunk, score in results:
    print(f"BM25 Score: {score:.4f}")
    print(f"Content: {chunk['page_content']}")
```

---

## 🆘 遇到問題？

查看詳細指南：[PYSERINI_SETUP.md](./PYSERINI_SETUP.md)

常見問題：
- ✓ Java 已安裝 (版本 23.0.1)
- ⚠️ 如果報錯 "pyserini not found"：運行 `pip install pyserini`
- ⚠️ 如果索引建立失敗：檢查磁碟空間是否足夠

---

## 📚 相關文件

- `pyserini_retriever.py` - Pyserini 實作
- `retriever.py` - 原有的 BM25Okapi 實作（保留）
- `compare_retrievers.py` - 性能比較工具
- `PYSERINI_SETUP.md` - 詳細安裝指南

祝使用順利！🎉
