# Retrieval results: recursive chunker

51 answerable questions from eval/questions.jsonl; embedding `sentence-transformers/all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-50 hybrid candidates. A hit counts when it overlaps >= 50% of a gold quote's characters.

| mode | recall@1 | recall@5 | recall@10 | MRR@10 | ms/query (CPU) |
|---|---|---|---|---|---|
| dense | 0.275 | 0.588 | 0.647 | 0.405 | 13.7 |
| bm25 | 0.490 | 0.716 | 0.833 | 0.596 | 1.5 |
| hybrid | 0.333 | 0.627 | 0.745 | 0.478 | 16.7 |
| hybrid_rerank | 0.480 | 0.716 | 0.853 | 0.606 | 1291.7 |
