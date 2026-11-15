"""The service must not serialise requests: with a connection pool, /health answers while an
/ask is busy in the (slow) reranker, and two slow requests overlap instead of queueing."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import FakeEmbedder, fake_rerank, seed
from fastapi.testclient import TestClient

from rag import db
from rag.api import create_app
from rag.generation import MockProvider
from rag.retrieval import Retriever
from rag.service import Platform

pytestmark = pytest.mark.integration

SLOW = 0.8  # seconds the fake reranker takes, standing in for the CPU cross-encoder


@pytest.fixture(scope="module")
def slow_platform(conn, db_url):
    emb = FakeEmbedder()
    seed(conn, emb)
    started = threading.Event()

    def slow_rerank(q, ps):
        started.set()
        time.sleep(SLOW)
        return fake_rerank(q, ps)

    pool = db.make_pool(db_url, min_size=2, max_size=4)
    r = Retriever(pool, "recursive", emb, rerank_fn=slow_rerank)
    p = Platform(pool, r, MockProvider(), abstain_threshold=0.5)
    p.started = started
    r.search("warm", "bm25", 1)  # build the BM25 index outside the timed section
    yield p
    pool.close()


def test_health_does_not_wait_for_ask(slow_platform):
    with TestClient(create_app(slow_platform)) as client, ThreadPoolExecutor(1) as ex:
        slow_platform.started.clear()
        fut = ex.submit(client.post, "/ask", json={"question": "What does core inflation exclude?"})
        assert slow_platform.started.wait(5)  # /ask is now inside the reranker
        t0 = time.perf_counter()
        r = client.get("/health")
        dt = time.perf_counter() - t0
        assert r.status_code == 200
        assert not fut.done(), "the /ask finished first, so the test proved nothing"
        assert dt < SLOW / 2, f"/health took {dt:.2f}s: it waited for /ask"
        assert fut.result().status_code == 200


def test_two_slow_searches_overlap(slow_platform):
    params = {"q": "core inflation", "mode": "hybrid_rerank", "k": 2}
    with TestClient(create_app(slow_platform)) as client, ThreadPoolExecutor(2) as ex:
        t0 = time.perf_counter()
        rs = list(ex.map(lambda _: client.get("/search", params=params), range(2)))
        wall = time.perf_counter() - t0
    assert all(r.status_code == 200 for r in rs)
    # Serialised would be >= 2 * SLOW; overlapped is ~SLOW plus overhead.
    assert wall < 1.5 * SLOW, f"two searches took {wall:.2f}s: they were serialised"


def test_single_connection_source_is_reentrant(conn):
    src = db.SingleConnection(conn)
    with src.connection() as a, src.connection() as b:  # nested borrow must not deadlock
        assert a is b and a.execute("SELECT 1").fetchone()[0] == 1
