"""Shared fixtures: one Testcontainers pgvector Postgres per session, a fresh database per
test module, a deterministic fake embedder/reranker and a tiny seeded corpus.

Fake models keep the integration tests to Docker only: no model download.
"""

import itertools
import os
import zlib

import numpy as np
import pytest

from rag import db
from rag.chunking import recursive
from rag.parse import Document
from rag.retrieval import tokenize

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


def fake_rerank(query, passages):
    """Fraction of query terms present in the passage, in [0, 1]."""
    q = set(tokenize(query))
    return [len(q & set(tokenize(p))) / max(1, len(q)) for p in passages]


DOCS = {
    "d-inflation": [
        "Inflation rose because the central bank kept policy rates low.",
        "Core inflation excludes food and energy prices.",
    ],
    "d-solar": ["Rooftop solar panels and home batteries change household electricity bills."],
    "d-debt": ["Hidden sovereign debt revelations raise bond spreads for years."],
}


def seed(conn, emb, docs=DOCS):
    """Migrate and ingest `docs` into the 'recursive' collection with the fake embedder."""
    db.migrate(conn)
    for doc_id, pages in docs.items():
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


@pytest.fixture(scope="session")
def pg_container():
    tc_pg = pytest.importorskip("testcontainers.postgres")
    try:
        container = tc_pg.PostgresContainer("pgvector/pgvector:pg16", driver=None)
        container.start()
    except Exception as exc:  # Docker not available
        if os.environ.get("RAG_REQUIRE_DOCKER"):  # CI: a silent skip would hide the suite
            raise
        pytest.skip(f"Docker unavailable: {exc}")
    yield container
    container.stop()


_db_counter = itertools.count()


@pytest.fixture(scope="module")
def db_url(pg_container):
    """URL of a brand-new, empty database (one per test module)."""
    admin_url = pg_container.get_connection_url()
    name = f"t{next(_db_counter)}"
    with db.connect(admin_url) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    return admin_url.rsplit("/", 1)[0] + f"/{name}"


@pytest.fixture(scope="module")
def conn(db_url):
    c = db.connect(db_url)
    yield c
    c.close()
