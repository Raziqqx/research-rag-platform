"""MCP server exposing the research corpus to any MCP client (official `mcp` SDK, v2 API).

    python -m rag.mcp_server                 # stdio (Claude Desktop, Claude Code, MCP Inspector)
    python -m rag.mcp_server --http 8001     # Streamable HTTP at http://localhost:8001/mcp

Tools: search, get_document, summarise, compare_reports. Handlers are async; the blocking
work (Postgres, CPU models) runs in a worker thread so the server's event loop stays free.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import anyio
from mcp.server.mcpserver import MCPServer

from rag.schemas import (
    AnswerModel,
    CompareResponse,
    DocumentResponse,
    Mode,
    Passage,
    SearchResponse,
    SummaryResponse,
)
from rag.service import Platform


def build_server(get_platform: Callable[[], Platform]) -> MCPServer:
    """`get_platform` is called on first use, so listing tools doesn't load any models."""
    server = MCPServer(
        "research-rag",
        instructions="Search and read openly licensed economics/finance research papers. "
        "Cite passages as [doc_id:page].",
    )
    state: dict[str, Platform] = {}

    async def run(fn_name: str, *args):
        def call():
            if "p" not in state:
                state["p"] = get_platform()
            return getattr(state["p"], fn_name)(*args)

        return await anyio.to_thread.run_sync(call)

    @server.tool()
    async def search(query: str, mode: Mode = "hybrid_rerank", k: int = 5) -> SearchResponse:
        """Search the corpus. mode: dense | bm25 | hybrid | hybrid_rerank (best quality)."""
        k = max(1, min(k, 50))
        hits = await run("search", query, mode, k)
        return SearchResponse(query=query, mode=mode, k=k, passages=[Passage.of(h) for h in hits])

    @server.tool()
    async def get_document(doc_id: str, page: int | None = None) -> DocumentResponse:
        """Metadata and page texts of one document; pass `page` (1-based) to get only that page."""
        doc = await run("get_document", doc_id)
        if doc is None:
            raise ValueError(f"document {doc_id!r} not found")
        pages = doc.pages
        if page is not None:
            if not 1 <= page <= len(pages):
                raise ValueError(f"page must be in 1..{len(pages)}")
            pages = [pages[page - 1]]
        return DocumentResponse(**doc.meta, pages=pages)

    @server.tool()
    async def summarise(doc_id: str, n_sentences: int = 5) -> SummaryResponse:
        """Cited summary of one document (extractive with the mock provider)."""
        ans = await run("summarise", doc_id, max(1, min(n_sentences, 10)))
        if ans is None:
            raise ValueError(f"document {doc_id!r} not found")
        return SummaryResponse(doc_id=doc_id, **AnswerModel.of(ans).model_dump())

    @server.tool()
    async def compare_reports(doc_a: str, doc_b: str, query: str, k: int = 3) -> CompareResponse:
        """Side-by-side key passages from two documents on the same query."""
        cmp = await run("compare", doc_a, doc_b, query, max(1, min(k, 10)))
        if cmp is None:
            raise ValueError(f"unknown document: {doc_a!r} or {doc_b!r}")
        return CompareResponse(
            query=query,
            doc_a=doc_a,
            doc_b=doc_b,
            passages_a=[Passage.of(h) for h in cmp.a],
            passages_b=[Passage.of(h) for h in cmp.b],
        )

    return server


def main() -> None:
    from rag.api import build_platform_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, metavar="PORT", help="serve Streamable HTTP on PORT")
    args = ap.parse_args()
    server = build_server(build_platform_from_env)
    if args.http:
        server.run("streamable-http", port=args.http)
    else:
        server.run()


if __name__ == "__main__":
    main()
