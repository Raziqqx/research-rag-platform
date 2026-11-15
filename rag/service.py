"""The platform facade shared by the FastAPI service and the MCP server.

Both front ends call the same `Platform` methods, so search, answers, summaries and comparisons
behave identically whichever way a client connects.
"""

from __future__ import annotations

import bisect
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import psycopg
from rank_bm25 import BM25Okapi

from rag.generation import (
    NOT_FOUND,
    Answer,
    Citation,
    Provider,
    check_citations,
    generate_answer,
    split_sentences,
)
from rag.parse import PAGE_SEP
from rag.retrieval import _COLS, Hit, Retriever, tokenize

# Calibrated on the eval set: see scripts/calibrate_abstain.py and results/abstention.md.
DEFAULT_ABSTAIN_THRESHOLD = -1.31  # same default as Settings


@dataclass
class DocumentView:
    meta: dict
    pages: list[str]


@dataclass
class Comparison:
    query: str
    a: list[Hit]
    b: list[Hit]


class Platform:
    def __init__(
        self,
        conn: psycopg.Connection,
        retriever: Retriever,
        provider: Provider,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    ):
        self.conn = conn
        self.retriever = retriever
        self.provider = provider
        self.threshold = abstain_threshold
        # One psycopg connection is shared; requests are serialised. The reranker is CPU-bound
        # anyway, so a pool would mostly add contention (run C revisits this).
        self.lock = threading.Lock()
        self._page_starts: dict[str, list[int]] = {}

    # ---- helpers
    def page_of(self, doc_id: str, offset: int) -> int:
        """1-based page of a character offset in a document's canonical text."""
        starts = self._page_starts.get(doc_id)
        if starts is None:
            lens = self.conn.execute(
                "SELECT length(text) FROM pages WHERE doc_id=%s ORDER BY page_no", (doc_id,)
            ).fetchall()
            starts, pos = [], 0
            for (n,) in lens:
                starts.append(pos)
                pos += n + len(PAGE_SEP)
            self._page_starts[doc_id] = starts
        return max(1, bisect.bisect_right(starts, offset))

    def _doc_chunks(self, doc_id: str) -> list[Hit]:
        rows = self.conn.execute(
            f"SELECT {_COLS} FROM chunks WHERE doc_id=%s AND strategy=%s AND embedding_model=%s "
            "ORDER BY chunk_index",
            (doc_id, self.retriever.strategy, self.retriever.embedder.model_name),
        ).fetchall()
        return [Hit(*r, 0.0) for r in rows]

    # ---- public operations
    def health(self) -> dict:
        with self.lock:
            n = self.conn.execute(
                "SELECT count(*) FROM chunks WHERE strategy=%s AND embedding_model=%s",
                (self.retriever.strategy, self.retriever.embedder.model_name),
            ).fetchone()[0]
        return {"chunks": n, "strategy": self.retriever.strategy, "provider": self.provider.name}

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 10) -> list[Hit]:
        with self.lock:
            return self.retriever.search(query, mode, k)

    def ask(self, question: str, k: int = 5) -> tuple[Answer, list[Hit]]:
        """Always retrieves with hybrid_rerank: abstention is gated on the reranker score."""
        with self.lock:
            hits = self.retriever.search(question, "hybrid_rerank", k)
            ans = generate_answer(question, hits, self.provider, self.threshold, self.page_of)
        return ans, hits

    def get_document(self, doc_id: str) -> DocumentView | None:
        with self.lock:
            cur = self.conn.execute(
                "SELECT doc_id, title, source, doc_type, topic, authors, published, url, licence, "
                "n_pages FROM documents WHERE doc_id=%s",
                (doc_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            meta = dict(zip([d.name for d in cur.description], row, strict=True))
            if meta["published"] is not None:
                meta["published"] = meta["published"].isoformat()
            pages = [
                r[0]
                for r in self.conn.execute(
                    "SELECT text FROM pages WHERE doc_id=%s ORDER BY page_no", (doc_id,)
                )
            ]
        return DocumentView(meta, pages)

    def summarise(self, doc_id: str, n_sentences: int = 5) -> Answer | None:
        """Extractive summary (Luhn-style): the sentences whose terms are most frequent in the
        document, in document order, each cited. With the Anthropic provider the selected
        passages are handed to the LLM to write the summary, and its citations are validated."""
        with self.lock:
            chunks = self._doc_chunks(doc_id)
            if not chunks:
                return None
            tf = Counter(t for h in chunks for t in tokenize(h.text))
            top_tf = max(tf.values())
            cands = []  # (score, chunk index, start, end)
            for ci, h in enumerate(chunks):
                for s, e in split_sentences(h.text):
                    toks = tokenize(h.text[s:e])
                    if not 8 <= len(toks) <= 60:  # skip headings, tables, run-on blocks
                        continue
                    score = sum(tf[t] for t in set(toks)) / (top_tf * len(set(toks)) ** 0.5)
                    cands.append((score, ci, s, e))
            best = sorted(sorted(cands, key=lambda c: -c[0])[:n_sentences], key=lambda c: c[1:3])
            if self.provider.name != "mock-extractive":
                used = sorted({c[1] for c in best})
                passages = [chunks[i] for i in used]
                raw = self.provider.answer(
                    "Summarise the main question, method and findings of this document.",
                    passages,
                    self.page_of,
                )
                chk = check_citations(raw, passages)
                return Answer(
                    chk.text,
                    chk.text == NOT_FOUND,
                    self.provider.name,
                    None,
                    chk.valid,
                    chk.invalid,
                    chk.unsupported_sentences,
                )
            parts, cites = [], []
            for _, ci, s, e in best:
                h = chunks[ci]
                page = self.page_of(doc_id, h.char_start + s)
                parts.append(f"{' '.join(h.text[s:e].split())} [{doc_id}:{page}]")
                if Citation(doc_id, page) not in cites:
                    cites.append(Citation(doc_id, page))
        return Answer(" ".join(parts) or NOT_FOUND, not parts, "mock-extractive", None, cites)

    def compare(self, doc_a: str, doc_b: str, query: str, k: int = 3) -> Comparison | None:
        """Key passages from each document for the same query, side by side.

        Within one document: BM25 over that document's chunks picks 20 candidates, which the
        cross-encoder reranks (the same scorer /ask uses), so both sides are ranked alike."""
        with self.lock:
            sides = []
            for doc_id in (doc_a, doc_b):
                chunks = self._doc_chunks(doc_id)
                if not chunks:
                    return None
                sides.append(self._rank_in_doc(chunks, query, k))
        return Comparison(query, *sides)

    def _rank_in_doc(self, chunks: Sequence[Hit], query: str, k: int) -> list[Hit]:
        bm = BM25Okapi([tokenize(h.text) for h in chunks]).get_scores(tokenize(query))
        cand = [chunks[i] for i in np.argsort(-bm, kind="stable")[:20]]
        scores = np.asarray(self.retriever.rerank_fn(query, [h.text for h in cand]))
        order = np.argsort(-scores, kind="stable")[:k]
        return [replace(cand[i], score=float(scores[i])) for i in order]
