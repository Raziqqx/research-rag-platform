"""FastAPI service: /search, /ask, /documents/{id}, /health.

    uvicorn rag.api:app --port 8000          # OpenAPI docs at /docs

`create_app(platform)` takes a ready Platform (tests inject one on a Testcontainers database
with fake models); the module-level `app` builds one from environment settings at startup.
Endpoints are plain `def`, so FastAPI runs them in its threadpool and the CPU-bound reranker
doesn't block the event loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from rag.schemas import (
    AnswerModel,
    AskRequest,
    AskResponse,
    DocumentResponse,
    Health,
    Mode,
    Passage,
    SearchResponse,
)
from rag.service import Platform

log = logging.getLogger("rag.api")


def build_platform_from_env() -> Platform:
    """Real models, real database, provider chosen by ANTHROPIC_API_KEY / RAG_LLM_PROVIDER."""
    from rag import db
    from rag.config import get_settings
    from rag.embed import Embedder
    from rag.generation import get_provider
    from rag.retrieval import Retriever

    s = get_settings()
    conn = db.connect(s.database_url)
    retriever = Retriever(conn, s.strategy, Embedder(s.embedding_model), s.reranker_model)
    return Platform(conn, retriever, get_provider(), s.abstain_threshold)


def create_app(
    platform: Platform | None = None, factory: Callable[[], Platform] = build_platform_from_env
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.platform = platform or factory()
        yield

    app = FastAPI(
        title="Research RAG Platform",
        version="0.2.0",
        description="Hybrid search and cited answers over openly licensed economics and "
        "finance papers.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
        log.info(
            "%s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code, ms
        )
        return response

    def plat(request: Request) -> Platform:
        return request.app.state.platform

    @app.get("/health", response_model=Health)
    def health(request: Request):
        try:
            h = plat(request).health()
        except Exception as exc:  # DB down or schema missing: report, don't crash
            log.warning("health check failed: %s", exc)
            body = Health(status="degraded", db=False, chunks=0, strategy="?", provider="?")
            return JSONResponse(body.model_dump(), status_code=503)
        status = "ok" if h["chunks"] > 0 else "degraded"
        return Health(status=status, db=True, **h)

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=1000),
        mode: Mode = "hybrid_rerank",
        k: int = Query(10, ge=1, le=50),
    ):
        hits = plat(request).search(q, mode, k)
        return SearchResponse(query=q, mode=mode, k=k, passages=[Passage.of(h) for h in hits])

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest):
        ans, hits = plat(request).ask(body.question, body.k)
        return AskResponse(
            question=body.question,
            passages=[Passage.of(h) for h in hits],
            **AnswerModel.of(ans).model_dump(),
        )

    @app.get("/documents/{doc_id}", response_model=DocumentResponse)
    def get_document(request: Request, doc_id: str):
        doc = plat(request).get_document(doc_id)
        if doc is None:
            raise HTTPException(404, f"document {doc_id!r} not found")
        return DocumentResponse(**doc.meta, pages=doc.pages)

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
