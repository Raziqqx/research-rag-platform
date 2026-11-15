"""Pick the abstention threshold on the top cross-encoder score (hybrid_rerank, recursive).

    python scripts/calibrate_abstain.py

For every eval question (answerable and unanswerable) records the best reranker score, then
picks the threshold that maximises balanced accuracy (mean of: answerable questions answered,
unanswerable questions abstained). Writes results/abstention.json and results/abstention.md.
Only 7 questions are unanswerable, so this is a calibration, not a held-out evaluation; run C
scores abstention end to end.
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import db  # noqa: E402
from rag.config import REPO_ROOT, get_settings  # noqa: E402
from rag.embed import Embedder  # noqa: E402
from rag.evalset import load_questions  # noqa: E402
from rag.parse import Document  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Threshold t (answer iff score >= t) maximising balanced accuracy; midpoint between
    adjacent observed scores. Returns (t, balanced accuracy)."""
    xs = sorted(set(pos + neg))
    cands = [xs[0] - 1.0] + [(a + b) / 2 for a, b in zip(xs, xs[1:], strict=False)] + [xs[-1] + 1]
    best = (cands[0], -1.0)
    for t in cands:
        bacc = (sum(p >= t for p in pos) / len(pos) + sum(n < t for n in neg) / len(neg)) / 2
        if bacc > best[1]:
            best = (t, bacc)
    return best


def main() -> None:
    s = get_settings()
    conn = db.connect(s.database_url)

    @cache
    def get_doc(doc_id: str) -> Document:
        pages = [
            r[0]
            for r in conn.execute(
                "SELECT text FROM pages WHERE doc_id=%s ORDER BY page_no", (doc_id,)
            )
        ]
        return Document(doc_id, pages)

    questions = load_questions(get_doc)
    r = Retriever(conn, "recursive", Embedder(s.embedding_model), s.reranker_model)
    rows = []
    for q in questions:
        top = r.search(q.question, "hybrid_rerank", 1)[0].score
        rows.append({"id": q.id, "answerable": q.answerable, "top_score": round(top, 4)})
    pos = [x["top_score"] for x in rows if x["answerable"]]
    neg = [x["top_score"] for x in rows if not x["answerable"]]
    t, bacc = best_threshold(pos, neg)
    t = round(t, 2)
    out = REPO_ROOT / "results"
    (out / "abstention.json").write_text(json.dumps({"threshold": t, "rows": rows}, indent=1))
    answered = sum(p >= t for p in pos)
    abstained = sum(n < t for n in neg)
    (out / "abstention.md").write_text(
        "# Abstention threshold calibration (hybrid_rerank, recursive collection)\n\n"
        "Produced by `scripts/calibrate_abstain.py`. Score = top cross-encoder logit.\n\n"
        f"- answerable questions: {len(pos)}, top score min/median/max = "
        f"{min(pos):.2f} / {sorted(pos)[len(pos) // 2]:.2f} / {max(pos):.2f}\n"
        f"- unanswerable questions: {len(neg)}, top score min/median/max = "
        f"{min(neg):.2f} / {sorted(neg)[len(neg) // 2]:.2f} / {max(neg):.2f}\n"
        f"- chosen threshold: **{t}** (balanced accuracy {bacc:.3f}): answers "
        f"{answered}/{len(pos)} answerable, abstains on {abstained}/{len(neg)} unanswerable\n"
        "\nCalibrated and reported on the same 58 questions (no held-out split; 7 negatives).\n"
    )
    print(
        f"threshold={t} bacc={bacc:.3f} answered={answered}/{len(pos)} "
        f"abstained={abstained}/{len(neg)}"
    )


if __name__ == "__main__":
    main()
