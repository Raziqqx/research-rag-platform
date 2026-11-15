# Retrieval ablations

Produced by `scripts/eval_ablation.py` on 51 answerable questions (eval/questions.jsonl). Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-50 hybrid candidates. A hit counts when it overlaps >= 50% of a gold quote's characters. CPU: `x86_64`; ms/query is end-to-end wall time including query embedding, measured on unweighted runs only (see script docstring).

| chunker | embedding | mode | RRF w (dense:bm25) | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|---|---|---|
| fixed | all-MiniLM-L6-v2 | dense | - | 0.245 | 0.500 | 0.618 | 0.359 | 14.6 |
| fixed | all-MiniLM-L6-v2 | bm25 | - | 0.451 | 0.755 | 0.804 | 0.588 | 1.8 |
| fixed | all-MiniLM-L6-v2 | hybrid | 1:1 | 0.363 | 0.627 | 0.735 | 0.490 | 18.5 |
| fixed | all-MiniLM-L6-v2 | hybrid_rerank | 1:1 | 0.578 | 0.794 | 0.853 | 0.672 | 1492.0 |
| recursive | all-MiniLM-L6-v2 | dense | - | 0.275 | 0.588 | 0.647 | 0.405 | 13.3 |
| recursive | all-MiniLM-L6-v2 | bm25 | - | 0.490 | 0.716 | 0.833 | 0.596 | 1.8 |
| recursive | all-MiniLM-L6-v2 | hybrid | 1:1 | 0.333 | 0.627 | 0.745 | 0.478 | 17.2 |
| recursive | all-MiniLM-L6-v2 | hybrid_rerank | 1:1 | 0.480 | 0.716 | 0.853 | 0.606 | 1322.2 |
| recursive | all-MiniLM-L6-v2 | hybrid | 0.25:1 | 0.490 | 0.686 | 0.794 | 0.585 | - |
| recursive | all-MiniLM-L6-v2 | hybrid_rerank | 0.25:1 | 0.480 | 0.735 | 0.863 | 0.612 | - |
| recursive | all-MiniLM-L6-v2 | hybrid | 0.5:1 | 0.392 | 0.686 | 0.775 | 0.527 | - |
| recursive | all-MiniLM-L6-v2 | hybrid_rerank | 0.5:1 | 0.480 | 0.735 | 0.863 | 0.612 | - |
| recursive | all-MiniLM-L6-v2 | hybrid | 2:1 | 0.333 | 0.608 | 0.725 | 0.467 | - |
| recursive | all-MiniLM-L6-v2 | hybrid_rerank | 2:1 | 0.480 | 0.716 | 0.853 | 0.606 | - |
| semantic | all-MiniLM-L6-v2 | dense | - | 0.294 | 0.549 | 0.667 | 0.406 | 14.8 |
| semantic | all-MiniLM-L6-v2 | bm25 | - | 0.490 | 0.686 | 0.794 | 0.595 | 2.4 |
| semantic | all-MiniLM-L6-v2 | hybrid | 1:1 | 0.431 | 0.637 | 0.745 | 0.528 | 18.3 |
| semantic | all-MiniLM-L6-v2 | hybrid_rerank | 1:1 | 0.480 | 0.696 | 0.824 | 0.597 | 1003.2 |
| recursive | bge-small-en-v1.5 | dense | - | 0.402 | 0.627 | 0.706 | 0.500 | 19.0 |
| recursive | bge-small-en-v1.5 | bm25 | - | 0.490 | 0.716 | 0.833 | 0.596 | 1.7 |
| recursive | bge-small-en-v1.5 | hybrid | 1:1 | 0.461 | 0.706 | 0.794 | 0.582 | 23.3 |
| recursive | bge-small-en-v1.5 | hybrid_rerank | 1:1 | 0.480 | 0.716 | 0.873 | 0.608 | 1318.9 |
| recursive | bge-small-en-v1.5 | hybrid | 0.25:1 | 0.451 | 0.725 | 0.833 | 0.568 | - |
| recursive | bge-small-en-v1.5 | hybrid_rerank | 0.25:1 | 0.480 | 0.735 | 0.863 | 0.612 | - |
| recursive | bge-small-en-v1.5 | hybrid | 0.5:1 | 0.441 | 0.706 | 0.814 | 0.565 | - |
| recursive | bge-small-en-v1.5 | hybrid_rerank | 0.5:1 | 0.480 | 0.735 | 0.863 | 0.612 | - |
| recursive | bge-small-en-v1.5 | hybrid | 2:1 | 0.441 | 0.686 | 0.755 | 0.557 | - |
| recursive | bge-small-en-v1.5 | hybrid_rerank | 2:1 | 0.480 | 0.716 | 0.853 | 0.607 | - |

## Best configuration (in-sample)

`fixed|all-MiniLM-L6-v2|hybrid_rerank|w=1:1`: recall@5 0.794, recall@10 0.853, MRR@10 0.672. Picked by MRR@10 (ties: recall@10) over all 28 configs on the same questions, so it is optimistic.

**5-fold cross-validated selection** (choose the best config by MRR@10 on 4/5 of the questions, score it on the held-out 1/5, pool): MRR@10 = **0.672**. Configs chosen per fold: `fixed|all-MiniLM-L6-v2|hybrid_rerank|w=1:1`, `fixed|all-MiniLM-L6-v2|hybrid_rerank|w=1:1`, `fixed|all-MiniLM-L6-v2|hybrid_rerank|w=1:1`, `fixed|all-MiniLM-L6-v2|hybrid_rerank|w=1:1`, `fixed|all-MiniLM-L6-v2|hybrid_rerank|w=1:1`.

Best config by question type:

| type | n | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|
| conceptual | 15 | 0.667 | 0.933 | 0.933 | 0.772 |
| factual | 20 | 0.750 | 0.950 | 0.950 | 0.817 |
| multi_doc | 3 | 0.167 | 0.500 | 0.500 | 0.417 |
| numeric | 13 | 0.308 | 0.462 | 0.692 | 0.395 |

With 51 questions one question is ~2 recall points (~0.02), so differences under ~0.04 are within noise.
