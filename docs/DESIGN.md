# Design

The full decision log, with the reasoning for each choice, is in `../DECISIONS.md`. This page
covers the main decisions only. All numbers come from `results/`.

## Key decisions and trade-offs

| Decision | Why | Alternatives considered | Trade-off |
|---|---|---|---|
| World Bank OKR + arXiv q-fin corpus | Licences can be checked by script (DSpace `dc.rights.uri`; arXiv OAI record + abs page) | IMF working papers | Most IMF papers are IMF-copyright or CC BY-NC-ND, which doesn't allow redistribution |
| PDFs downloaded by script, not committed | Keeps the repo small; sha256 pinning makes runs reproducible | Committing the PDFs | Needs network access on first run |
| pymupdf "blocks" mode | Gives paragraph structure, which the recursive chunker needs | pypdf (one line per text line) | Math-heavy layouts still produce single-line blocks |
| Chunks as character offsets into the canonical text | Pages for citations come from offsets; one gold set scores every chunker | Gold labels by chunk id | Gold labels must be verbatim quotes |
| 1000-character chunks for all strategies | Fits MiniLM's 256-token window without silent truncation | Token-based sizes | Fixed size isn't tuned per model |
| BM25 in memory (rank_bm25) | True BM25 with IDF and length saturation | Postgres `ts_rank` (no IDF) | The index is per process and has to be rebuilt after ingest |
| One partial HNSW index per (strategy, model) | A global index filters after the graph walk and can return fewer than k rows | One global index | More indexes |
| RRF, k = 60 | Simple, uses ranks only, no score calibration | Linear score fusion | Weights both lists equally unless weighted |
| Abstention before generation, on the reranker score | No LLM call for questions the corpus can't answer | Asking the LLM to refuse | The threshold needs calibration (see limitations) |
| Strip invalid citations, flag unsupported sentences | The reader sees exactly what is grounded | Rejecting the whole answer | Answers can be partly flagged |
| Token-overlap faithfulness (>= 80 % of a sentence's content tokens in a cited passage) | Deterministic, no extra model, enough for an extractive mock | NLI cross-encoder | Misses meaning flips such as "rose" vs "fell" |
| One `Platform` facade for both API and MCP | The two interfaces can't drift apart | Separate implementations | None significant |
| psycopg_pool; connections held only around SQL | A 1.3 s rerank doesn't block `/health` | One connection behind a lock (run B) | Requests still share the CPU |
| Plain SQL migrations | No ORM, small schema | Alembic | Fewer migration features |
| OCI Always Free A1 VM running compose | 12 GB RAM, free with no expiry, multi-arch images | GCP e2-micro (1 GB RAM), AWS (time-limited credits) | Postgres runs in a container, so backups are our job |

## Baseline vs improved

The baseline is dense-only retrieval with all-MiniLM-L6-v2 on recursive chunks. Numbers are on
51 answerable questions, from `results/ablation_retrieval.md`.

| config | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| baseline: recursive, MiniLM, dense | 0.275 | 0.588 | 0.647 | 0.405 | 13.3 |
| recursive, MiniLM, bm25 | 0.490 | 0.716 | 0.833 | 0.596 | 1.8 |
| recursive, MiniLM, hybrid (RRF 1:1) | 0.333 | 0.627 | 0.745 | 0.478 | 17.2 |
| recursive, bge-small, hybrid (RRF 1:1) | 0.461 | 0.706 | 0.794 | 0.582 | 23.3 |
| **deployed**: recursive, MiniLM, hybrid_rerank | 0.480 | 0.716 | 0.853 | 0.606 | 1322.2 |
| **best (in-sample and CV)**: fixed, MiniLM, hybrid_rerank | 0.578 | 0.794 | 0.853 | 0.672 | 1492.0 |

What the ablation shows:
- **Plain RRF hurt compared with BM25** on MiniLM (MRR 0.478 vs 0.596). The dense list is much
  weaker, and RRF weights both lists equally. bge-small fixes most of this (0.582), and so does
  down-weighting dense to 0.25:1 (0.585).
- **After reranking, the embedding model barely matters.** All recursive variants land at MRR
  0.606-0.612. The cross-encoder dominates, so what counts is the candidate pool.
- **Fixed-size chunks win**, probably because the 150-character overlap splits fewer gold
  quotes across two chunks. 5-fold CV picked this config in all 5 folds (held-out MRR 0.672).
- **Cost:** BM25 takes about 2 ms per query and hybrid_rerank 1.0-1.5 s on CPU. BM25 alone is
  close to the deployed config at R@5 (both 0.716).
- **By question type** (best config), numeric questions are the weakest (R@10 0.692, MRR 0.395)
  and multi-document questions score R@10 0.500 on only n = 3.

## Why the deployed config isn't the CV-best one

The deployed config is recursive + hybrid_rerank. The abstention threshold (-1.31) was
calibrated on recursive's reranker scores, so switching chunkers without recalibrating would
break abstention. The MRR gap (0.066) is about 3 questions out of 51. Switching takes
`RAG_STRATEGY=fixed`, `make calibrate` and `scripts/eval_answers.py`. That is logged as the
next step rather than done blindly.

## Known limitations

- **Generation has only been measured with the extractive mock.** The `AnthropicProvider` is
  implemented but has never run (no API key). Faithfulness of 139/139 = 1.000 (control 0.079)
  shows that the citation plumbing is right. It is not a measure of LLM faithfulness.
- **Citation accuracy is 37/128 = 0.289.** The mock cites up to 3 passages, and usually only one
  holds the gold quote. 37/51 = 0.725 answers cite at least one gold page.
- **Abstention uses n = 7 unanswerable questions.** In-sample precision and recall are 1.000.
  Under leave-one-out they are 0.857 / 0.857, with thresholds ranging from -2.29 to -0.34.
- **The eval set is small and has one annotator.** It has 51 answerable questions (one question
  is about 0.02 recall) and leans towards abstracts and introductions. Recall is a lower bound
  because each question has only one gold quote.
- **Terraform has never been applied**, so the cloud-init boot path is untested on a real VM.
- **CI has never run on GitHub** (no remote). Its steps were run locally.
- BM25 is in memory per process, which is fine for this corpus but not for millions of chunks.
- The mock can pick a section heading as a "sentence".

## Next steps

1. Run the Anthropic provider and re-measure faithfulness with an NLI cross-encoder
   (e.g. `cross-encoder/nli-deberta-v3-small`) on the same sentence/passage pairs.
2. Switch to fixed chunks, recalibrate abstention and re-run the answer eval.
3. Write many more unanswerable questions, and calibrate on a held-out split.
4. Improve numeric questions: try stemming or number-aware tokenisation, and a larger rerank
   pool.
5. Apply the Terraform to a real OCI account, add a nightly `pg_dump` to Object Storage and a
   remote state backend.
6. Push to GitHub so CI runs. Move BM25 to `pg_search` or OpenSearch if the corpus grows.
