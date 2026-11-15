-- 001: documents, pages and chunks. Applied in filename order by rag/db.py (migrate()).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        text PRIMARY KEY,
    title         text NOT NULL,
    source        text NOT NULL,          -- taxonomy: publisher (World Bank OKR, arXiv)
    doc_type      text NOT NULL,          -- taxonomy: working_paper | preprint
    topic         text,                   -- taxonomy: arXiv category or OKR topic query
    authors       text[] NOT NULL DEFAULT '{}',
    published     date,
    url           text NOT NULL,
    licence       text NOT NULL,
    sha256        text NOT NULL,
    n_pages       int  NOT NULL
);

-- Cleaned page text, kept so citations and the document viewer can show the exact page.
CREATE TABLE IF NOT EXISTS pages (
    doc_id   text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_no  int  NOT NULL,               -- 1-based
    text     text NOT NULL,
    PRIMARY KEY (doc_id, page_no)
);

-- One row per (chunking strategy, embedding model, chunk). char offsets index into the
-- document's canonical text (pages joined by a blank line; see rag/parse.py).
CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    doc_id          text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    strategy        text NOT NULL,
    embedding_model text NOT NULL,
    chunk_index     int  NOT NULL,
    char_start      int  NOT NULL,
    char_end        int  NOT NULL,
    page_start      int  NOT NULL,
    page_end        int  NOT NULL,
    text            text NOT NULL,
    embedding       vector(384) NOT NULL,
    UNIQUE (doc_id, strategy, embedding_model, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (strategy, embedding_model);

CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
-- HNSW indexes are partial, one per (strategy, model) collection, and are created by
-- rag/db.py:ensure_hnsw_index() when a collection is first ingested.
