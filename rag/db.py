"""Postgres + pgvector access: migrations, writes and the HNSW index per collection."""

from __future__ import annotations

import contextlib
import re
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from rag.config import REPO_ROOT

MIGRATIONS_DIR = REPO_ROOT / "migrations"


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


class ConnectionSource(Protocol):
    """Anything that lends out a connection: psycopg_pool.ConnectionPool or SingleConnection."""

    def connection(self) -> contextlib.AbstractContextManager[psycopg.Connection]: ...


class SingleConnection:
    """One connection behind the pool interface, for scripts and tests.

    Borrowers are serialised by a re-entrant lock (a psycopg connection must not run two
    queries at once). The service uses a real pool instead; see make_pool.
    """

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        self._lock = threading.RLock()

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        with self._lock:
            yield self.conn


def as_source(conn_or_pool: psycopg.Connection | ConnectionSource) -> ConnectionSource:
    if isinstance(conn_or_pool, psycopg.Connection):
        return SingleConnection(conn_or_pool)
    return conn_or_pool


def _configure(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)


def make_pool(url: str, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    """Pool of autocommit connections with the pgvector type registered on each.

    Each request borrows a connection only for the duration of its queries, so /health never
    waits behind a slow /ask (the reranker runs with no connection held).
    """
    return ConnectionPool(
        url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True},
        configure=_configure,
        open=True,
        name="rag",
    )


def migrate(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every *.sql file not yet recorded in schema_migrations, in filename order."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    done = {r[0] for r in conn.execute("SELECT filename FROM schema_migrations")}
    applied = []
    for f in sorted(migrations_dir.glob("*.sql")):
        if f.name in done:
            continue
        with conn.transaction():
            conn.execute(f.read_text())
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (f.name,))
        applied.append(f.name)
    return applied


def collection_slug(strategy: str, model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", f"{strategy}_{model.split('/')[-1]}".lower()).strip("_")


def ensure_hnsw_index(conn: psycopg.Connection, strategy: str, model: str) -> None:
    """One partial HNSW index per (strategy, model) collection.

    A single global HNSW index would be post-filtered by the WHERE clause, so a query for one
    collection could get fewer than k rows back (the graph walk returns ef_search candidates
    from *all* collections first). A partial index whose predicate matches the query avoids that.
    Values are interpolated because DDL can't take bind parameters; both come from our own
    config and are checked here.
    """
    for v in (strategy, model):
        if not re.fullmatch(r"[\w./-]+", v):
            raise ValueError(f"unsafe collection name {v!r}")
    name = f"chunks_hnsw_{collection_slug(strategy, model)}"[:63]
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {name} ON chunks USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = 16, ef_construction = 64) "
        f"WHERE strategy = '{strategy}' AND embedding_model = '{model}'"
    )


def upsert_document(conn: psycopg.Connection, meta: dict, pages: list[str]) -> None:
    with conn.transaction():
        conn.execute(
            """INSERT INTO documents (doc_id, title, source, doc_type, topic, authors, published,
                                      url, licence, sha256, n_pages)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title,
                   sha256 = EXCLUDED.sha256, n_pages = EXCLUDED.n_pages""",
            (
                meta["doc_id"],
                meta["title"],
                meta["source"],
                meta["doc_type"],
                meta.get("topic_query"),
                meta.get("authors", []),
                meta.get("published") or None,
                meta["url"],
                meta["licence"],
                meta["sha256"],
                len(pages),
            ),
        )
        conn.execute("DELETE FROM pages WHERE doc_id = %s", (meta["doc_id"],))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO pages (doc_id, page_no, text) VALUES (%s, %s, %s)",
                [(meta["doc_id"], i + 1, t) for i, t in enumerate(pages)],
            )


def replace_chunks(
    conn: psycopg.Connection,
    doc_id: str,
    strategy: str,
    model: str,
    rows: Iterable[tuple[int, int, int, int, str, np.ndarray]],
) -> int:
    """Replace one document's chunks in a collection. rows: (start, end, p_start, p_end, text,
    embedding), in document order."""
    rows = list(rows)
    with conn.transaction():
        conn.execute(
            "DELETE FROM chunks WHERE doc_id=%s AND strategy=%s AND embedding_model=%s",
            (doc_id, strategy, model),
        )
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (doc_id, strategy, embedding_model, chunk_index, char_start,
                       char_end, page_start, page_end, text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(doc_id, strategy, model, i, *r) for i, r in enumerate(rows)],
            )
    return len(rows)
