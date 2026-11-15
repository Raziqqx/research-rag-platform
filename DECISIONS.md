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

## Generation and citations (run B)
- **Provider interface (`rag/generation.py`)**: `answer(question, passages, page_of) -> text`.
  The default is a deterministic **extractive mock**: for each of the top passages it picks
  the sentence covering the most question terms and cites it as `[doc_id:page]`, with the
  page computed from the sentence's character offset (not just the chunk's first page).
  `AnthropicProvider` (claude-opus-5-5, official SDK, effort `low`, server-side refusal
  fallback) is used only when `ANTHROPIC_API_KEY` is set (or `RAG_LLM_PROVIDER=anthropic`).
  It has not been run in this repo: no key is available, so every answer quoted here comes
  from the mock.
- **Abstention is gated before generation** on the top cross-encoder score, so the LLM is not
  called for questions the corpus can't answer. Threshold **-1.31** (env
  `RAG_ABSTAIN_THRESHOLD`) from `scripts/calibrate_abstain.py` → `results/abstention.md`: on
  the recursive collection the top score separates the 51 answerable questions (min -0.70)
  from the 7 unanswerable ones (max -1.91), and the threshold is the midpoint. Caveat:
  calibrated and measured on the same 58 questions with only 7 negatives; run C needs a
  held-out check. Only passages scoring above the threshold are shown to the provider.
- **/ask always retrieves with `hybrid_rerank`**, because abstention needs the cross-encoder
  score; the mode selector applies to `/search` only.
- **Citation validation**: a citation is valid only if a passage that was shown to the provider
  comes from that doc and spans that page. Invalid ones are **stripped** from the text and
  returned in `invalid_citations`; sentences left with no valid citation are **flagged** in
  `unsupported_sentences` (kept, so the reader sees what is unsupported). If no valid citation
  survives, the system abstains rather than return an ungrounded answer.
- **Summaries are extractive (Luhn-style)**: sentences scored by the document-frequency of
  their terms, 8–60 tokens, top n kept in document order and cited. With the Anthropic
  provider, the chunks holding those sentences go to the LLM and its citations are validated.
- **compare_reports**: per document, BM25 over that document's chunks picks 20 candidates and
  the cross-encoder reranks them, so both sides are scored by the same model as /ask.

## Service (run B)
- **One `Platform` facade (`rag/service.py`)** serves both FastAPI and MCP, so they can't drift.
- ~~One psycopg connection behind a lock~~ (run B) **replaced in run C by a `psycopg_pool`
  ConnectionPool** (`rag/db.py:make_pool`, size `RAG_DB_POOL_SIZE`, default 8). Every
  operation borrows a connection only around its SQL and returns it before model inference,
  so a 1.3 s reranker call no longer holds anything /health needs. Scripts and tests that pass
  a bare connection get it wrapped in `SingleConnection` (re-entrant lock, same
  `.connection()` interface). The lazy BM25 build is guarded by its own lock so concurrent
  first requests build it once. `tests/test_concurrency.py` checks, with a reranker that
  sleeps 0.8 s, that /health returns in < 0.4 s while /ask is mid-rerank and that two
  /search calls overlap (wall < 1.2 s rather than >= 1.6 s). Concurrent requests still share
  the CPU, so throughput on one machine is bounded by the cross-encoder; scaling out means
  more API replicas or a separate model server.
- **Timing logs**: an HTTP middleware logs `METHOD path -> status in X ms` and sets
  `X-Response-Time-Ms`. `/health` returns 503 `degraded` if the DB query fails, and
  `degraded` if the collection has no chunks.
- **Tests share one Testcontainers Postgres per session (`tests/conftest.py`)**, with a fresh
  database per test module and fake embedder/reranker, so they need Docker but no models.
  `Retriever` takes an injectable `rerank_fn` for this.

## MCP server (run B)
- **`mcp` 2.3 (`MCPServer`, the renamed FastMCP)**. Tools are async and run the blocking
  Platform calls with `anyio.to_thread.run_sync`; they return the same pydantic models as the
  API, so clients get structured content plus an output schema. The Platform is built on first
  tool call, so `list_tools` doesn't load models. Transports: stdio (default) and
  Streamable HTTP (`--http PORT`). Tests use the SDK's in-process `Client(server)`.

## Frontend and compose (run B)
- **Web: React 18 + TypeScript on Vite 5** (Node 18 on this machine), no UI library. Calls
  relative `/api/*`; the Vite dev proxy and the nginx config in the web image both strip the
  prefix and forward to the API, so the same build works in dev and in compose.
  One vitest + Testing Library component test covers citation parsing and the click-to-show
  passage behaviour.
- **Compose: db + api + web.** The API image installs CPU-only torch (image 2.49 GB, measured)
  and does not bake models in: `${RAG_DATA_DIR:-~/rag-data}/hf-cache` is mounted with
  `HF_HUB_OFFLINE=1`. The db service reuses the existing `pgdata` volume, so the already
  ingested corpus is served. `ANTHROPIC_API_KEY` is passed through; empty means mock.
- **End-to-end check (2026-11-16, `docker compose up -d --build --wait`, all three healthy):**
  `GET :8000/health` → 200 `{"status":"ok","chunks":1544,"strategy":"recursive",
  "provider":"mock-extractive"}`. `POST :8080/api/ask` (through nginx) "How did Ecuador's
  hidden debt come to light in 2018?" → 200 in 1.18 s, not abstained, top score 6.25, first
  sentence is the gold quote cited `[wb-10986-45628:6]` (the gold page), 0 invalid citations.
  "Who won the 1998 football World Cup?" → 200 in 2.0 s, abstained, top score -8.84.
  Then `docker compose stop api web`.
- **Known limitations seen in that run:** the mock can extract a section heading as a
  "sentence" (e.g. "Identification of hidden debt revelations"); and a `/health` request that
  arrived during an `/ask` waited ~1.1 s for the shared-connection lock (logged). Run C: give
  health its own connection or a pool, and filter heading-like fragments in the mock.

## Terraform (run C)
- **Oracle Cloud (OCI) Always Free: one Ampere A1 VM (2 OCPU / 12 GB, 100 GB boot volume)
  running the compose stack**, in `infra/terraform/`. Compared options: GCP's free e2-micro has
  1 GB RAM, too small for torch plus two models; AWS's free tier for new accounts is now
  time-limited credits, and RDS/Cloud SQL are not free beyond that. OCI A1 (up to 4 OCPU /
  24 GB) is free with no expiry and arm64 wheels exist for CPU torch, and the pgvector image is
  multi-arch.
- **Postgres is a container on the VM, not a managed service**: OCI's free managed database is
  Oracle Autonomous DB, which has no pgvector. Trade-off: backups and failover are on us
  (next step: a nightly `pg_dump` to Always Free Object Storage).
- Network: VCN + public subnet; the security list opens 22 to `admin_cidr` only (validated,
  0.0.0.0/0 rejected) and 80 to the world. Postgres and the API port are never exposed; nginx
  serves the UI and proxies `/api`. Secrets (`postgres_password`, `anthropic_api_key`) are
  `sensitive` variables written to a 0600 `.env` by cloud-init; `terraform.tfvars` is
  git-ignored and only `terraform.tfvars.example` is committed. State is local (no backend);
  a shared deployment would use an OCI Object Storage backend.
- Cloud-init installs Docker, clones the repo, runs the `ingest` compose job (fetch +
  sha256-check the PDFs, embed), then `docker compose up -d --wait`.
- **Checked with `make tf-check`** (`terraform fmt -check`, `init -backend=false`,
  `validate` in the `hashicorp/terraform:1.9.8` image): passes. **Never applied**: no OCI
  account or credentials were used, so the boot script is untested on a real VM.
- WSL note: `docker pull` failed through the Docker Desktop credential helper; pulling with
  an empty `DOCKER_CONFIG` works for public images.

## Compose, reproduce and CI (run C)
- **Compose profiles for one-off jobs**: `ingest` (fetch PDFs with sha256 check, ingest the
  three chunkers on MiniLM plus recursive on bge-small; the only service allowed to download
  models) and `eval` (runs the eval scripts with `eval/` and `results/` mounted). They reuse
  the API image (`research-rag-api`), so the job and the server run the same code. Every
  long-running service has a healthcheck; web waits for a healthy API, the API for a healthy
  DB. The Postgres password and host port are env-configurable (`POSTGRES_PASSWORD`,
  `DB_PORT`), defaulting to the local-dev values.
- **Verified (2026-11-16):** `docker compose up -d --build --wait` -> db, api, web all
  `healthy`. A `/health` sent 0.3 s into a 2.5 s `/ask` returned in 0.06 s (the pool fix, end
  to end). `docker compose --profile ingest run --rm ingest python scripts/fetch_corpus.py`
  -> "22/22 PDFs verified"; the eval profile loads all 58 questions in the container. The
  full ingest/eval jobs were not re-run in containers (same scripts as the host run that
  produced results/). Containers stopped afterwards (db left running as found).
- **`make reproduce`** regenerates every file in `results/` on the host venv: db -> corpus ->
  ingest (3 chunkers) -> ingest-bge -> eval-retrieval -> calibrate -> eval (ablation +
  answers). `make eval` is the run C eval only.
- **CI (`.github/workflows/ci.yml`)**: three jobs, no secrets. Python: CPU-only torch, ruff
  check + format check, pytest with `HF_HUB_OFFLINE=1` (a test that tried to download a model
  would fail) and `RAG_REQUIRE_DOCKER=1` (conftest raises instead of skipping when Docker is
  missing, so the Testcontainers suite can't silently vanish). The fixture corpus is the
  four in-code documents in `tests/conftest.py`, not the PDFs. Web: Node 20, `npm ci`,
  build, vitest. Terraform: fmt -check, init -backend=false, validate. Not run on GitHub from
  here (no remote); every step was run locally with the same commands.

## Full evaluation (run C)
- **`make eval` = `scripts/eval_ablation.py` + `scripts/eval_answers.py`**; outputs are CSV +
  md in `results/` (`ablation_retrieval.*`, `ablation_retrieval_per_question.csv`,
  `answer_quality.*`).
- **Ablations are retrieval-only** (recall@1/5/10, MRR@10, ms/query): 3 chunkers x 4 modes on
  MiniLM; bge-small-en-v1.5 on the recursive chunker (ingested as a second collection, same
  schema; no query instruction prefix, which BGE v1.5 documents as optional); weighted RRF
  with dense:bm25 = 0.25, 0.5, 2 (plus 1) on both recursive collections. Weighted runs reuse
  memoised cross-encoder scores per (collection, query, passage), so only the unweighted runs
  report latency.
- **Choosing the "best config" is itself fitted on the eval set**, so the md also reports a
  5-fold cross-validated selection (pick by MRR@10 on 4/5, score on the held-out 1/5).
- **Results (2026-11-16):** best is **fixed-size chunks (1000 chars, 150 overlap) +
  hybrid_rerank on MiniLM: R@5 0.794, R@10 0.853, MRR@10 0.672**; CV selection picked the
  same config in all 5 folds, held-out MRR@10 0.672. Recursive + hybrid_rerank (the deployed
  default) is 0.716 / 0.853 / 0.606. bge-small lifts dense alone (MRR 0.405 -> 0.500, R@10
  0.647 -> 0.706) and fixes plain RRF (MRR 0.478 -> 0.582), but after reranking all
  recursive variants land at MRR 0.606-0.612: the cross-encoder dominates, and the
  candidate pool is what matters. Down-weighting MiniLM in RRF (0.25:1) helps hybrid without
  rerank but changes hybrid_rerank by < 1 question. Latency: BM25 ~2 ms, dense ~13-19 ms,
  hybrid ~17-23 ms, hybrid_rerank 1.0-1.5 s per query on CPU.
- **Default left at recursive** despite fixed scoring higher: the abstention threshold was
  calibrated on recursive's reranker scores, and the MRR gap (0.066) is ~3 questions at
  n = 51. Switching means `RAG_STRATEGY=fixed` plus re-running `make calibrate` and
  `eval_answers.py`; recorded as the next step rather than done blindly. Likely reason fixed
  wins: its 150-char overlap means a gold quote is less often split across two chunks.
- **Faithfulness = fraction of answer sentences whose content tokens are >= 80 % present in a
  passage the sentence cites** (`rag.metrics.support`). Chosen over NLI because it is
  deterministic, needs no extra model and runs in microseconds on CPU, and because the only
  provider that has been run is the extractive mock, whose sentences are copied verbatim, so
  an entailment model would add cost without discriminating anything. Its blind spot:
  paraphrase that flips meaning ("rose" vs "fell") passes. A control scores the same
  sentences against another question's passages to show the check can fail. **Result:
  139/139 = 1.000 on the mock (control 0.079)**: this validates citation plumbing, not LLM
  faithfulness. With a real LLM, the next step is an NLI cross-encoder
  (e.g. `cross-encoder/nli-deberta-v3-small`) on the same sentence/passage pairs.
- **Citation accuracy**: a citation is correct when its page holds part of a gold quote.
  37/128 citations = 0.289 (the mock cites up to 3 passages and usually only one holds the
  quote); 37/51 = 0.725 answers cite at least one gold page.
- **Abstention, held out by leave-one-out**: each question is judged by a threshold fitted
  (balanced accuracy) on the other 57. LOO precision 0.857, recall 0.857 (6/7 unanswerable
  abstained, 1/51 answerable wrongly abstained), against 1.000 / 1.000 in-sample at the
  deployed -1.31. LOO thresholds ranged -2.29..-0.34, i.e. the threshold is unstable with 7
  negatives; more unanswerable questions are the fix.
