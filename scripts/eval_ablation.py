"""Retrieval ablations: chunker x mode, a second embedding model, and weighted RRF.

    python scripts/eval_ablation.py

Needs the collections from `make ingest` (fixed/recursive/semantic x all-MiniLM-L6-v2) and
`make ingest-bge` (recursive x bge-small-en-v1.5). Scores the 51 answerable questions with the
same rules as scripts/eval_retrieval.py (recall@1/5/10, MRR@10). Writes:

  results/ablation_retrieval.csv               one row per config (summary)
  results/ablation_retrieval_per_question.csv  one row per config x question
  results/ablation_retrieval.md                tables + the cross-validated choice of config

Latency is wall time per query on this CPU, measured on the unweighted runs with nothing
cached. Weighted-RRF runs reuse cross-encoder scores already computed for the same
(collection, question, passage), so their latency is not reported (it equals the unweighted
hybrid_rerank latency up to noise: same models, same 50 candidates).
"""

from __future__ import annotations

import csv
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import db  # noqa: E402
from rag.config import REPO_ROOT, get_settings  # noqa: E402
from rag.embed import Embedder, cross_encoder_scores  # noqa: E402
from rag.evalset import doc_loader, load_questions  # noqa: E402
from rag.metrics import match_matrix, mean, recall_at_k, reciprocal_rank  # noqa: E402
from rag.retrieval import MODES, Retriever  # noqa: E402

KS = (1, 5, 10)
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"
# (strategy, embedding model): the chunker grid on MiniLM, plus bge-small on recursive.
COLLECTIONS = [
    ("fixed", MINILM),
    ("recursive", MINILM),
    ("semantic", MINILM),
    ("recursive", BGE),
]
# Weighted RRF: dense weight (bm25 weight fixed at 1.0), tried on both recursive collections.
DENSE_WEIGHTS = (0.25, 0.5, 2.0)
N_FOLDS = 5
METRICS = [f"recall@{k}" for k in KS] + ["mrr@10"]


def score(retriever: Retriever, questions, mode: str) -> tuple[list[dict], float]:
    rows, t0 = [], time.perf_counter()
    for q in questions:
        hits = retriever.search(q.question, mode, max(KS))
        m = match_matrix([(h.doc_id, h.char_start, h.char_end) for h in hits], q.gold)
        row = {f"recall@{k}": recall_at_k(m, len(q.gold), k) for k in KS}
        row["mrr@10"] = reciprocal_rank(m, 10)
        rows.append({"id": q.id, "type": q.type, **row})
    return rows, (time.perf_counter() - t0) / len(questions) * 1000


def cv_select(per_q: dict[str, list[dict]], metric: str = "mrr@10") -> tuple[float, list[str]]:
    """Honest estimate of "pick the best config": for each fold, choose the config with the
    best mean `metric` on the other folds, then score it on the held-out fold. Returns the
    pooled held-out mean and the config chosen in each fold."""
    n = len(next(iter(per_q.values())))
    held, chosen = [], []
    for f in range(N_FOLDS):
        train = [i for i in range(n) if i % N_FOLDS != f]
        test = [i for i in range(n) if i % N_FOLDS == f]
        best = max(per_q, key=lambda c: mean([per_q[c][i][metric] for i in train]))
        chosen.append(best)
        held += [per_q[best][i][metric] for i in test]
    return mean(held), chosen


def main() -> None:
    s = get_settings()
    conn = db.connect(s.database_url)
    questions = [q for q in load_questions(doc_loader(conn)) if q.answerable]
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    summary: list[dict] = []
    per_q: dict[str, list[dict]] = {}
    for strategy, model in COLLECTIONS:
        emb = Embedder(model)
        memo: dict[tuple[str, str], float] = {}  # (query, passage) -> cross-encoder score

        def rerank(q: str, ps: list[str], memo=memo) -> list[float]:
            todo = [p for p in dict.fromkeys(ps) if (q, p) not in memo]
            for p, v in zip(todo, cross_encoder_scores(s.reranker_model, q, todo), strict=True):
                memo[(q, p)] = float(v)
            return [memo[(q, p)] for p in ps]

        runs = [(mode, (1.0, 1.0), True) for mode in MODES]
        if strategy == "recursive":
            runs += [
                (mode, (w, 1.0), False)
                for w in DENSE_WEIGHTS
                for mode in ("hybrid", "hybrid_rerank")
            ]
        for mode, weights, timed in runs:
            r = Retriever(
                conn, strategy, emb, s.reranker_model, rerank_fn=rerank, rrf_weights=weights
            )
            r.search("warm-up", "bm25", 1)  # build BM25 + load models outside the timing
            emb(["warm-up"])
            if timed and mode == "hybrid_rerank":
                cross_encoder_scores(s.reranker_model, "warm-up", ["x"])
            rows, ms = score(r, questions, mode)
            name = f"{strategy}|{model.split('/')[-1]}|{mode}|w={weights[0]:g}:{weights[1]:g}"
            per_q[name] = rows
            summ = {
                "strategy": strategy,
                "embedding": model.split("/")[-1],
                "mode": mode,
                "rrf_weights": f"{weights[0]:g}:{weights[1]:g}" if "hybrid" in mode else "",
                **{m: round(mean([x[m] for x in rows]), 3) for m in METRICS},
                "ms_per_query": round(ms, 1) if timed else "",
            }
            summary.append(summ)
            print(name, {m: summ[m] for m in METRICS}, summ["ms_per_query"])

    with (out / "ablation_retrieval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with (out / "ablation_retrieval_per_question.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "id", "type", *METRICS])
        for name, rows in per_q.items():
            for x in rows:
                w.writerow([name, x["id"], x["type"], *(x[m] for m in METRICS)])

    best = max(summary, key=lambda r: (r["mrr@10"], r["recall@10"]))
    cv_mrr, chosen = cv_select(per_q)
    by_type: dict[str, dict[str, float]] = {}
    best_name = (
        f"{best['strategy']}|{best['embedding']}|{best['mode']}|w={best['rrf_weights'] or '1:1'}"
    )
    for t in sorted({x["type"] for x in per_q[best_name]}):
        xs = [x for x in per_q[best_name] if x["type"] == t]
        by_type[t] = {"n": len(xs), **{m: round(mean([x[m] for x in xs]), 3) for m in METRICS}}

    hdr = (
        "| chunker | embedding | mode | RRF w (dense:bm25) | R@1 | R@5 | R@10 | MRR@10 | ms/query |"
    )
    lines = [
        "# Retrieval ablations",
        "",
        f"Produced by `scripts/eval_ablation.py` on {len(questions)} answerable questions "
        f"(eval/questions.jsonl). Reranker `{s.reranker_model}` over the top-50 hybrid "
        "candidates. A hit counts when it overlaps >= 50% of a gold quote's characters. "
        f"CPU: `{platform.processor() or platform.machine()}`; ms/query is end-to-end wall time "
        "including query embedding, measured on unweighted runs only (see script docstring).",
        "",
        hdr,
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['strategy']} | {r['embedding']} | {r['mode']} | {r['rrf_weights'] or '-'} | "
            f"{r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
            f"{r['mrr@10']:.3f} | {r['ms_per_query'] or '-'} |"
        )
    lines += [
        "",
        "## Best configuration (in-sample)",
        "",
        f"`{best_name}`: recall@5 {best['recall@5']:.3f}, recall@10 {best['recall@10']:.3f}, "
        f"MRR@10 {best['mrr@10']:.3f}. Picked by MRR@10 (ties: recall@10) over all "
        f"{len(summary)} configs on the same questions, so it is optimistic.",
        "",
        f"**{N_FOLDS}-fold cross-validated selection** (choose the best config by MRR@10 on 4/5 "
        "of the questions, score it on the held-out 1/5, pool): "
        f"MRR@10 = **{cv_mrr:.3f}**. Configs chosen per fold: "
        + ", ".join(f"`{c}`" for c in chosen)
        + ".",
        "",
        "Best config by question type:",
        "",
        "| type | n | R@1 | R@5 | R@10 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for t, v in by_type.items():
        lines.append(
            f"| {t} | {v['n']} | {v['recall@1']:.3f} | {v['recall@5']:.3f} | "
            f"{v['recall@10']:.3f} | {v['mrr@10']:.3f} |"
        )
    lines += [
        "",
        "With 51 questions one question is ~2 recall points (~0.02), so differences under ~0.04 "
        "are within noise.",
    ]
    (out / "ablation_retrieval.md").write_text("\n".join(lines) + "\n")
    print("best:", best_name, "cv mrr:", round(cv_mrr, 3), chosen)


if __name__ == "__main__":
    main()
