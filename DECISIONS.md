# DECISIONS.md

Decisions taken without asking, with the reasoning. Newest at the bottom of each section.

## Corpus
- **Sources: World Bank OKR working papers + arXiv q-fin preprints.** IMF working papers were
  not used: most carry IMF copyright or CC BY-NC-ND, which doesn't allow redistribution of
  derivatives. The OKR exposes licence metadata through its DSpace REST API, and arXiv shows
  the licence on each abs page, so both can be verified by script.
- **Licence verification is automated and strict.** OKR: the item record's `dc.rights.uri`
  must be CC BY 3.0 IGO or CC BY 4.0 IGO (BY-NC items were seen and dropped). arXiv: the OAI
  record must say CC BY 4.0 *and* the abs page must link the CC BY 4.0 deed. The evidence URL
  is stored per document in `corpus/manifest.json`.
- **22 documents, 582 pages**, 8–45 pages each (keeps the eval readable and ingest under
  ~4 minutes on CPU). Topics: one OKR paper per topic query; at most two arXiv papers per q-fin
  sub-category (arXiv submissions 2025-03-01..20).
- **PDFs live under `$RAG_PDF_DIR` (default `~/rag-data/pdfs`)**, not in git. The manifest pins
  the arXiv version (e.g. `v2`) and stores a sha256; `scripts/fetch_corpus.py` re-downloads and
  fails on a hash mismatch.

## Parsing and chunking
- **pymupdf "blocks" mode** (layout paragraphs in reading order) instead of pypdf: pypdf
  returns one line per text line with no paragraph structure, which the recursive chunker
  needs. Cleaning: NFKC (folds ﬁ/ﬃ ligatures), remove soft hyphens and line-end hyphenation,
  drop bare page-number blocks.
- **Chunks are stored as character offsets into a canonical document text** (pages joined by a
  blank line). Page numbers for citations come from the offsets, and the eval matches gold
  quotes by character overlap, so all strategies are scored by the same rule.
- **Size: 1000 characters** for all three strategies (~200–250 word pieces), so chunks fit
  all-MiniLM-L6-v2's 256-token window instead of being silently truncated. Fixed-size uses
  150 characters of overlap; recursive and semantic tile the text without overlap.
- **Semantic chunker:** each sentence is embedded with one neighbour either side; a break goes
  where the cosine distance between consecutive windows exceeds the document's 90th
  percentile; groups < 200 chars merge forward and groups > 1000 chars are split recursively.
  It reuses the ingest embedding model, so no extra model is needed.

## Storage
- **Plain SQL migrations** (`migrations/*.sql`, applied in order and recorded in
  `schema_migrations`) instead of Alembic: there is no ORM, and the schema is small.
- **Chunks are keyed by (strategy, embedding_model)** so several chunkers and embedding
  models coexist in one table for the run C ablations. The column is `vector(384)`; both
  candidate models (all-MiniLM-L6-v2, bge-small-en-v1.5) are 384-d.
- **One partial HNSW index per (strategy, model)** rather than one global index. With a global
  index, pgvector applies the `WHERE strategy = ...` filter after the graph walk, so a query
  can return fewer than k rows. The integration test checks the planner uses the partial index.
  `hnsw.ef_search` is set per query to max(64, 2k).
- **Postgres data uses a Docker named volume.** On Docker Desktop that lives in its VM disk; the
  database is 42 MB (measured), so this doesn't conflict with keeping large data off C:.
- Host port **5433** to avoid clashing with other local Postgres instances.

## Retrieval
- **BM25 via `rank_bm25` (Okapi, k1=1.5, b=0.75) in memory, not Postgres `ts_rank`.**
  `ts_rank`/`ts_rank_cd` aren't BM25: they have no IDF term and no length saturation, so a
  common word counts as much as a rare one. The corpus is a few thousand chunks per
  collection, so building the index on first use takes 0.13 s (measured, recursive collection). Trade-off: the
  index is per process and must be rebuilt after ingest; at millions of chunks this would move
  to a Postgres BM25 extension (ParadeDB `pg_search`) or OpenSearch.
- Tokenizer: lowercase alphanumeric tokens (keeping `2.5`, `u.s`), small stopword list, no
  stemming. Stemming is a candidate ablation.
- **RRF with k = 60** (Cormack et al. 2009) over the top 50 of each list; ties broken by best
  single-list rank, then first appearance, so results are deterministic.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top 50 hybrid candidates.
- **Embedding model default: `sentence-transformers/all-MiniLM-L6-v2`** (fast on CPU, 384-d);
  bge-small-en-v1.5 is the planned comparison in run C.

## Evaluation
- See `eval/METHOD.md`. Gold = (doc, page, verbatim quote); a hit covers a gold passage when it
  overlaps >= 50 % of the quote's characters. recall@k counts each gold passage once; MRR is
  cut at 10. Unanswerable questions are excluded from recall/MRR.
- CPU-only torch from the PyTorch CPU wheel index; venv and HF cache under `~/rag-data`.
- **Default search mode: `hybrid_rerank`.** On the recursive collection it has the best
  recall@10 (0.853) and MRR@10 (0.606) in `results/retrieval_recursive.md`, though BM25 alone
  is within noise at recall@5 (both 0.716) and ~860x cheaper on CPU (1.5 ms vs 1.29 s per
  query). Plain RRF *hurts* versus BM25 here because the MiniLM dense list is much weaker
  (recall@10 0.647) and RRF weights both lists equally. Likely causes, to test in run C: gold
  quotes favour exact terms (numbers, names), and MiniLM is a general-purpose model that
  wasn't trained on finance text. Candidates: bge-small, weighted RRF, and a bigger rerank pool.
