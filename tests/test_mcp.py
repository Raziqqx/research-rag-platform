"""MCP server tools through the SDK's in-process client, on a Testcontainers database."""

import pytest
from conftest import FakeEmbedder, fake_rerank, seed
from mcp import Client

from rag.generation import MockProvider
from rag.mcp_server import build_server
from rag.retrieval import Retriever
from rag.service import Platform

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def server(conn):
    emb = FakeEmbedder()
    seed(conn, emb)
    r = Retriever(conn, "recursive", emb, rerank_fn=fake_rerank)
    return build_server(lambda: Platform(conn, r, MockProvider(), abstain_threshold=0.5))


async def call(server, name, args):
    async with Client(server) as c:
        res = await c.call_tool(name, args)
    return res


async def test_lists_four_tools(server):
    async with Client(server) as c:
        tools = {t.name for t in (await c.list_tools()).tools}
    assert tools == {"search", "get_document", "summarise", "compare_reports"}


async def test_search_tool(server):
    res = await call(server, "search", {"query": "rooftop solar batteries", "mode": "bm25"})
    assert not res.is_error
    assert res.structured_content["passages"][0]["doc_id"] == "d-solar"


async def test_get_document_tool_single_page(server):
    res = await call(server, "get_document", {"doc_id": "d-inflation", "page": 2})
    assert res.structured_content["pages"] == ["Core inflation excludes food and energy prices."]
    bad = await call(server, "get_document", {"doc_id": "d-inflation", "page": 9})
    assert bad.is_error


async def test_summarise_tool_cites_its_document(server):
    res = await call(server, "summarise", {"doc_id": "d-debt"})
    body = res.structured_content
    assert not body["abstained"] and "[d-debt:1]" in body["answer"]


async def test_compare_reports_tool(server):
    res = await call(
        server,
        "compare_reports",
        {"doc_a": "d-inflation", "doc_b": "d-solar", "query": "prices", "k": 1},
    )
    body = res.structured_content
    assert body["passages_a"][0]["doc_id"] == "d-inflation"
    assert body["passages_b"][0]["doc_id"] == "d-solar"
    missing = await call(server, "compare_reports", {"doc_a": "x", "doc_b": "y", "query": "q"})
    assert missing.is_error
