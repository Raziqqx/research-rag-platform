"""End-to-end store + retrieve against a real pgvector Postgres (Testcontainers).

Uses a deterministic fake 384-d embedder so the test needs Docker but no model download.
"""

import zlib

import numpy as np
import pytest

from rag import db
from rag.chunking import recursive
from rag.parse import Document
from rag.retrieval import Retriever

pytestmark = pytest.mark.integration
tc_pg = pytest.importorskip("testcontainers.postgres")

MODEL = "fake/hash-384"


class FakeEmbedder:
    model_name = MODEL

    def __call__(self, texts):
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.strip(".,").encode()) % 384] += 1
        out += 1e-6
        return out / np.linalg.norm(out, axis=1, keepdims=True)


DOCS = {
    "d-inflation": [
        "Inflation rose because the central bank kept policy rates low.",
        "Core inflation excludes food and energy prices.",
    ],
    "d-solar": ["Rooftop solar panels and home batteries change household electricity bills."],
    "d-debt": ["Hidden sovereign debt revelations raise bond spreads for years."],
}


@pytest.fixture(scope="module")
def conn():
    try:
        container = tc_pg.PostgresContainer("pgvector/pgvector:pg16", driver=None)
        container.start()
    except Exception as exc:  # Docker not available
        pytest.skip(f"Docker unavailable: {exc}")
    c = db.connect(container.get_connection_url())
    yield c
    c.close()
    container.stop()


def ingest(conn, emb):
    assert db.migrate(conn) == ["001_init.sql"]
    assert db.migrate(conn) == []  # idempotent
    for doc_id, pages in DOCS.items():
        meta = {
            "doc_id": doc_id,
            "title": doc_id,
            "source": "test",
            "doc_type": "t",
            "url": "http://x",
            "licence": "CC BY 4.0",
            "sha256": "0",
        }
        db.upsert_document(conn, meta, pages)
        doc = Document(doc_id, pages)
        spans = recursive(doc.text, max_chars=80)
        texts = [doc.text[s.start : s.end] for s in spans]
        rows = [
            (s.start, s.end, doc.page_of(s.start), doc.page_of(s.end - 1), t, v)
            for s, t, v in zip(spans, texts, emb(texts), strict=True)
        ]
        db.replace_chunks(conn, doc_id, "recursive", MODEL, rows)
    db.ensure_hnsw_index(conn, "recursive", MODEL)


def test_store_and_search(conn):
    emb = FakeEmbedder()
    ingest(conn, emb)
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert n == 4

    r = Retriever(conn, "recursive", emb)
    for mode in ("dense", "bm25", "hybrid"):
        hits = r.search("core inflation food energy", mode, k=3)
        assert hits[0].doc_id == "d-inflation", mode
        assert hits[0].page_start == 2  # page numbers survive the round trip
    assert r.search("solar batteries", "dense", 1)[0].doc_id == "d-solar"

    # Re-ingesting a document replaces, not duplicates, its chunks.
    ingest_again = [(0, 10, 1, 1, "replaced x", emb(["replaced x"])[0])]
    db.replace_chunks(conn, "d-solar", "recursive", MODEL, ingest_again)
    assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id='d-solar'").fetchone()[0] == 1


def test_partial_hnsw_index_is_used(conn):
    emb = FakeEmbedder()
    qv = emb(["inflation"])[0]
    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")  # tiny table: force the planner's hand
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT id FROM chunks WHERE strategy='recursive' AND embedding_model=%s "
                "ORDER BY embedding <=> %s LIMIT 3",
                (MODEL, qv),
            )
        )
    assert "chunks_hnsw_recursive_hash_384" in plan


def test_unsafe_collection_name_rejected(conn):
    with pytest.raises(ValueError):
        db.ensure_hnsw_index(conn, "recursive'; DROP TABLE chunks; --", MODEL)
