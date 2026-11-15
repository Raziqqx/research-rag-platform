"""Retrieval-only evaluation: recall@k and MRR for each search mode on one chunk collection.

    python scripts/eval_retrieval.py --strategy recursive

Writes results/retrieval_<strategy>.json (per-question detail) and .md (summary table).
Unanswerable questions are excluded from recall/MRR; they are scored in run C (abstention).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import db  # noqa: E402
from rag.config import REPO_ROOT, get_settings  # noqa: E402
from rag.embed import Embedder  # noqa: E402
from rag.evalset import load_questions  # noqa: E402
from rag.metrics import match_matrix, mean, recall_at_k, reciprocal_rank  # noqa: E402
from rag.parse import Document  # noqa: E402
from rag.retrieval import MODES, Retriever  # noqa: E402

KS = (1, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = ap.parse_args()

    s = get_settings()
    conn = db.connect(s.database_url)

    @cache
    def get_doc(doc_id: str) -> Document:
        # Rebuild the canonical text from stored pages so offsets match ingest exactly.
        pages = [
            r[0]
            for r in conn.execute(
                "SELECT text FROM pages WHERE doc_id=%s ORDER BY page_no", (doc_id,)
            )
        ]
        if not pages:
            raise LookupError(f"document {doc_id} is not ingested")
        return Document(doc_id, pages)

    questions = [q for q in load_questions(get_doc) if q.answerable]
    retriever = Retriever(conn, args.strategy, Embedder(s.embedding_model), s.reranker_model)
    retriever.search("warm-up", "hybrid_rerank", 1)  # load models / BM25 before timing

    summary, detail = {}, []
    for mode in args.modes:
        rows, t0 = [], time.perf_counter()
        for q in questions:
            hits = retriever.search(q.question, mode, max(KS))
            m = match_matrix([(h.doc_id, h.char_start, h.char_end) for h in hits], q.gold)
            row = {f"recall@{k}": recall_at_k(m, len(q.gold), k) for k in KS}
            row["mrr@10"] = reciprocal_rank(m, 10)
            rows.append(row)
            detail.append(
                {
                    "mode": mode,
                    "id": q.id,
                    "type": q.type,
                    **row,
                    "top_docs": [h.doc_id for h in hits[:5]],
                }
            )
        latency = (time.perf_counter() - t0) / len(questions) * 1000
        summary[mode] = {key: round(mean([r[key] for r in rows]), 3) for key in rows[0]}
        summary[mode]["ms_per_query"] = round(latency, 1)
        print(mode, summary[mode])

    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)
    meta = {
        "strategy": args.strategy,
        "embedding_model": s.embedding_model,
        "reranker_model": s.reranker_model,
        "n_questions_scored": len(questions),
        "cpu": platform.processor() or platform.machine(),
    }
    (out / f"retrieval_{args.strategy}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "per_question": detail}, indent=1) + "\n"
    )
    lines = [
        f"# Retrieval results: {args.strategy} chunker",
        "",
        f"{len(questions)} answerable questions from eval/questions.jsonl; embedding "
        f"`{s.embedding_model}`, reranker `{s.reranker_model}` over the top-50 hybrid "
        "candidates. A hit counts when it overlaps >= 50% of a gold quote's characters.",
        "",
        "| mode | recall@1 | recall@5 | recall@10 | MRR@10 | ms/query (CPU) |",
        "|---|---|---|---|---|---|",
    ]
    for mode, r in summary.items():
        lines.append(
            f"| {mode} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | "
            f"{r['recall@10']:.3f} | {r['mrr@10']:.3f} | {r['ms_per_query']} |"
        )
    (out / f"retrieval_{args.strategy}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote results/retrieval_{args.strategy}.md")


if __name__ == "__main__":
    main()
