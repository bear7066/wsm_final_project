python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input dragonball_dataset/collections \
  --index searcher_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw