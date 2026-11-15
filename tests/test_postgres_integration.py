"""End-to-end store + retrieve against a real pgvector Postgres (Testcontainers).

Uses a deterministic fake 384-d embedder so the test needs Docker but no model download.
"""

import pytest
from conftest import MODEL, FakeEmbedder
from conftest import seed as _seed

from rag import db
from rag.retrieval import Retriever

pytestmark = pytest.mark.integration


def ingest(conn, emb):
    assert db.migrate(conn) == ["001_init.sql"]
    assert db.migrate(conn) == []  # idempotent
    _seed(conn, emb)


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
