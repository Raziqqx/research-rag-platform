"""FastAPI endpoints against a real pgvector database (Testcontainers) with fake models."""

import pytest
from conftest import FakeEmbedder, fake_rerank, seed
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.generation import NOT_FOUND, MockProvider
from rag.retrieval import Retriever
from rag.service import Platform

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def platform(conn):
    emb = FakeEmbedder()
    seed(conn, emb)
    r = Retriever(conn, "recursive", emb, rerank_fn=fake_rerank)
    return Platform(conn, r, MockProvider(), abstain_threshold=0.5)


@pytest.fixture(scope="module")
def client(platform):
    with TestClient(create_app(platform)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["db"] and body["chunks"] == 4
    assert body["provider"] == "mock-extractive"
    assert float(r.headers["X-Response-Time-Ms"]) >= 0


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_search_modes(client, mode):
    r = client.get("/search", params={"q": "core inflation food energy", "mode": mode, "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == mode and len(body["passages"]) <= 2
    top = body["passages"][0]
    assert top["doc_id"] == "d-inflation" and top["page_start"] == 2


def test_search_validation(client):
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_ask_answers_with_valid_citation(client):
    r = client.post("/ask", json={"question": "What does core inflation exclude?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"]
    assert "[d-inflation:2]" in body["answer"]
    assert body["citations"] == [{"doc_id": "d-inflation", "page": 2}]
    assert body["invalid_citations"] == [] and body["unsupported_sentences"] == []
    # every citation is backed by a returned passage
    for c in body["citations"]:
        assert any(
            p["doc_id"] == c["doc_id"] and p["page_start"] <= c["page"] <= p["page_end"]
            for p in body["passages"]
        )


def test_ask_abstains_when_corpus_lacks_answer(client):
    body = client.post("/ask", json={"question": "Who won the 1998 football World Cup?"}).json()
    assert body["abstained"] and body["answer"] == NOT_FOUND and body["citations"] == []


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "?"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_get_document(client):
    r = client.get("/documents/d-inflation")
    assert r.status_code == 200
    body = r.json()
    assert body["n_pages"] == 2 and body["pages"][1].startswith("Core inflation")
    assert body["licence"] == "CC BY 4.0"
    assert client.get("/documents/nope").status_code == 404


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/ask", "/documents/{doc_id}", "/health"} <= set(paths)


def test_health_reports_db_failure(platform):
    class Broken(Platform):
        def health(self):
            raise RuntimeError("connection refused")

    broken = Broken(platform.conn, platform.retriever, platform.provider)
    with TestClient(create_app(broken)) as c:
        r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


def test_summarise_and_compare(platform):
    s = platform.summarise("d-inflation", 2)
    assert s is not None and not s.abstained
    assert {c.doc_id for c in s.citations} == {"d-inflation"}
    cmp = platform.compare("d-inflation", "d-debt", "inflation spreads", k=1)
    assert cmp.a[0].doc_id == "d-inflation" and cmp.b[0].doc_id == "d-debt"
    assert platform.summarise("nope") is None and platform.compare("nope", "d-debt", "x") is None
