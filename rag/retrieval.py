"""Selectable retrieval modes over one chunk collection (strategy x embedding model).

dense          pgvector cosine KNN on the collection's partial HNSW index
bm25           Okapi BM25 (rank_bm25) over the collection's chunk texts, built in memory
hybrid         Reciprocal Rank Fusion of dense and bm25 candidate lists
hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg
from rank_bm25 import BM25Okapi

from rag.db import ConnectionSource, as_source
from rag.embed import Embedder, cross_encoder_scores

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60  # Cormack et al. (2009); damps the influence of the very top ranks
CANDIDATES = 50  # depth of each list fed into fusion and the reranker

# Short English stopword list: enough to stop "the"/"of" dominating BM25 on long questions.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to "
    "was were what when where which who why will with does do did than then there these those "
    "their they into over under about after before between can could would should".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def rrf(
    rankings: Sequence[Sequence[Hashable]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of w_i / (k + rank_d), rank from 1.

    `weights` (default all 1.0) is weighted RRF: w_i scales list i's contribution, e.g. to
    trust a strong BM25 list more than a weak dense one.

    Ties are broken by the best single-list rank, then by first appearance, so output is
    deterministic.
    """
    score: dict[Hashable, float] = {}
    best: dict[Hashable, int] = {}
    order: dict[Hashable, int] = {}
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + w / (k + rank)
            best[item] = min(best.get(item, rank), rank)
            order.setdefault(item, len(order))
    return sorted(score.items(), key=lambda kv: (-kv[1], best[kv[0]], order[kv[0]]))


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_id: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str
    score: float


RerankFn = Callable[[str, list[str]], Sequence[float]]

_COLS = "id, doc_id, page_start, page_end, char_start, char_end, text"


class Retriever:
    def __init__(
        self,
        conn: psycopg.Connection | ConnectionSource,
        strategy: str,
        embedder: Embedder,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_fn: RerankFn | None = None,
        rrf_weights: tuple[float, float] = (1.0, 1.0),
    ):
        # A pool (service) or a single connection (scripts/tests). Connections are borrowed only
        # around queries, never across model inference.
        self.db = as_source(conn)
        self.strategy = strategy
        self.embedder = embedder
        self.reranker_model = reranker_model
        # (query, passages) -> scores. Injectable so API/MCP tests run without the model.
        self.rerank_fn: RerankFn = rerank_fn or (
            lambda q, ps: cross_encoder_scores(reranker_model, q, ps)
        )
        self.rrf_weights = rrf_weights  # (dense, bm25) for hybrid fusion
        self._bm25: BM25Okapi | None = None
        self._rows: list[tuple] = []
        self._bm25_lock = threading.Lock()  # build the in-memory index once, even under load

    # ---- dense
    def dense(self, query: str, k: int) -> list[Hit]:
        qv = self.embedder([query])[0]
        with self.db.connection() as conn, conn.transaction():
            # ef_search must be >= k or HNSW returns fewer than k rows.
            conn.execute(f"SET LOCAL hnsw.ef_search = {max(64, 2 * k)}")
            rows = conn.execute(
                f"""SELECT {_COLS}, 1 - (embedding <=> %s) AS sim FROM chunks
                    WHERE strategy = %s AND embedding_model = %s
                    ORDER BY embedding <=> %s LIMIT %s""",
                (qv, self.strategy, self.embedder.model_name, qv, k),
            ).fetchall()
        return [Hit(*r) for r in rows]

    # ---- bm25
    def _ensure_bm25(self) -> BM25Okapi:
        with self._bm25_lock:
            if self._bm25 is None:
                with self.db.connection() as conn:
                    rows = conn.execute(
                        f"SELECT {_COLS} FROM chunks WHERE strategy=%s AND embedding_model=%s "
                        "ORDER BY id",
                        (self.strategy, self.embedder.model_name),
                    ).fetchall()
                if not rows:
                    raise LookupError(f"no chunks for strategy {self.strategy!r}")
                self._rows = rows
                self._bm25 = BM25Okapi([tokenize(r[6]) for r in rows])
            return self._bm25

    def bm25(self, query: str, k: int) -> list[Hit]:
        scores = self._ensure_bm25().get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(*self._rows[i], float(scores[i])) for i in top if scores[i] > 0]

    # ---- fusion and reranking
    def hybrid(self, query: str, k: int) -> list[Hit]:
        lists = [self.dense(query, CANDIDATES), self.bm25(query, CANDIDATES)]
        by_id = {h.chunk_id: h for hits in lists for h in hits}
        fused = rrf([[h.chunk_id for h in hits] for hits in lists], weights=self.rrf_weights)
        return [_rescore(by_id[cid], s) for cid, s in fused[:k]]

    def hybrid_rerank(self, query: str, k: int) -> list[Hit]:
        cands = self.hybrid(query, CANDIDATES)
        scores = np.asarray(self.rerank_fn(query, [h.text for h in cands]))
        order = np.argsort(-scores, kind="stable")[:k]
        return [_rescore(cands[i], float(scores[i])) for i in order]

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 10) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        return getattr(self, mode)(query, k)


def _rescore(h: Hit, score: float) -> Hit:
    return Hit(
        h.chunk_id, h.doc_id, h.page_start, h.page_end, h.char_start, h.char_end, h.text, score
    )
