"""End-to-end answer evaluation: faithfulness, citation accuracy and abstention.

    python scripts/eval_answers.py

Runs every eval question (58: 51 answerable, 7 unanswerable) through Platform.ask on the
production config (recursive chunker, all-MiniLM-L6-v2, hybrid_rerank, k=5) with the provider
from the environment (the extractive mock unless ANTHROPIC_API_KEY is set). Writes
results/answer_quality.csv (per question) and results/answer_quality.md.

Faithfulness: an answer sentence is *supported* when at least SUPPORT_MIN of its content
tokens occur in a passage it cites (same doc, page inside the passage's span; see
rag.metrics.support). Sentences with no citation count as unsupported. A control re-scores
every answer against a different question's passages, to show the check can fail.

Citation accuracy: a citation [doc:page] is correct when the page holds (part of) a gold quote
for that question.

Abstention: positive class = "abstain". Reported at the deployed threshold (in-sample: it was
fitted on these same questions) and with leave-one-out thresholds (held out).
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import db  # noqa: E402
from rag.config import REPO_ROOT, get_settings  # noqa: E402
from rag.embed import Embedder  # noqa: E402
from rag.evalset import doc_loader, load_questions  # noqa: E402
from rag.generation import CITATION, get_provider  # noqa: E402
from rag.metrics import loo_abstain, mean, precision_recall, support  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402
from rag.service import Platform  # noqa: E402

SUPPORT_MIN = 0.8
# "sentence [d:p] [d:p]" -> (sentence, citations); the provider puts citations after a claim.
_CLAIM = re.compile(r"(.+?)((?:\s*\[[A-Za-z0-9][\w.\-]*:\d+\])+)\s*", re.S)


def claims(answer: str) -> list[tuple[str, list[tuple[str, int]]]]:
    out, pos = [], 0
    for m in _CLAIM.finditer(answer):
        pos = m.end()
        cites = [(d, int(p)) for d, p in CITATION.findall(m.group(2))]
        out.append((m.group(1).strip(), cites))
    rest = answer[pos:].strip()
    if rest:
        out.append((rest, []))  # trailing text with no citation
    return out


def sentence_support(sentence, cites, passages) -> float:
    cands = [
        h.text
        for h in passages
        for d, p in cites
        if h.doc_id == d and h.page_start <= p <= h.page_end
    ]
    return max((support(sentence, t) for t in cands), default=0.0)


def main() -> None:
    s = get_settings()
    conn = db.connect(s.database_url)
    get_doc = doc_loader(conn)
    questions = load_questions(get_doc)
    retriever = Retriever(conn, "recursive", Embedder(s.embedding_model), s.reranker_model)
    plat = Platform(conn, retriever, get_provider(), s.abstain_threshold)

    rows, shown = [], []
    for q in questions:
        ans, hits = plat.ask(q.question, k=5)
        usable = [h for h in hits if h.score >= s.abstain_threshold]
        gold_pages = {
            (g.doc_id, p)
            for g in q.gold
            for p in range(
                get_doc(g.doc_id).page_of(g.start), get_doc(g.doc_id).page_of(g.end - 1) + 1
            )
        }
        row = {
            "id": q.id,
            "type": q.type,
            "answerable": q.answerable,
            "top_score": round(hits[0].score, 3) if hits else "",
            "abstained": ans.abstained,
            "n_sentences": 0,
            "n_supported": 0,
            "faithfulness": "",
            "n_citations": 0,
            "n_citations_gold": 0,
            "gold_cited": "",
            "answer": ans.answer,
        }
        cl = [] if ans.abstained else claims(ans.answer)
        if cl:
            sup = [sentence_support(x, c, usable) >= SUPPORT_MIN for x, c in cl]
            cites = list(dict.fromkeys(c for _, cs in cl for c in cs))
            row |= {
                "n_sentences": len(cl),
                "n_supported": sum(sup),
                "faithfulness": round(sum(sup) / len(cl), 3),
                "n_citations": len(cites),
                "n_citations_gold": sum(c in gold_pages for c in cites),
                "gold_cited": any(c in gold_pages for c in cites) if q.answerable else "",
            }
        rows.append(row)
        shown.append((cl, usable))

    # Control: score each answer's claims against the *next* answered question's passages.
    answered = [i for i, r in enumerate(rows) if r["n_sentences"]]
    ctl = []
    for a, b in zip(answered, answered[1:] + answered[:1], strict=True):
        cl, _ = shown[a]
        other = shown[b][1]
        ctl += [max((support(x, h.text) for h in other), default=0.0) >= SUPPORT_MIN for x, _ in cl]

    out = REPO_ROOT / "results"
    with (out / "answer_quality.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ans_rows = [r for r in rows if r["answerable"] and r["n_sentences"]]
    n_sent = sum(r["n_sentences"] for r in rows)
    n_sup = sum(r["n_supported"] for r in rows)
    n_cit = sum(r["n_citations"] for r in ans_rows)
    n_cit_gold = sum(r["n_citations_gold"] for r in ans_rows)
    n_ans = sum(r["answerable"] for r in rows)

    actual = [not r["answerable"] for r in rows]
    p_in, r_in = precision_recall([r["abstained"] for r in rows], actual)
    scores = [r["top_score"] if r["top_score"] != "" else -1e9 for r in rows]
    loo = loo_abstain(scores, [r["answerable"] for r in rows])
    p_cv, r_cv = precision_recall([a for _, a in loo], actual)
    ts = sorted(t for t, _ in loo)
    fa = sum(r["abstained"] and r["answerable"] for r in rows)
    fa_cv = sum(a and r["answerable"] for (_, a), r in zip(loo, rows, strict=True))

    n_gc = sum(r["gold_cited"] is True for r in ans_rows)
    row_in = (
        f"| deployed {s.abstain_threshold} (fitted on these 58: in-sample) "
        f"| {p_in:.3f} | {r_in:.3f} | {fa}/{n_ans} |"
    )
    row_cv = (
        "| leave-one-out (each question judged by a threshold fitted on the other 57) "
        f"| **{p_cv:.3f}** | **{r_cv:.3f}** | {fa_cv}/{n_ans} |"
    )
    md = f"""# Answer quality (end to end)

Produced by `scripts/eval_answers.py`: all {len(rows)} eval questions through `Platform.ask`
(recursive chunker, all-MiniLM-L6-v2, hybrid_rerank, k = 5, abstention threshold
{s.abstain_threshold}). **Provider: `{plat.provider.name}`.** With the extractive mock every
answer sentence is copied from a retrieved passage, so faithfulness here mostly checks the
citation plumbing (right passage, right page), not an LLM's tendency to hallucinate; an LLM
provider must be re-measured with the same script.

## Faithfulness
Answer sentences supported by a passage they cite (>= {SUPPORT_MIN:.0%} of the sentence's
content tokens occur in that passage; uncited sentences count as unsupported):

- sentence level: **{n_sup}/{n_sent} = {n_sup / max(1, n_sent):.3f}**
- mean per answer: {mean([r["faithfulness"] for r in rows if r["n_sentences"]]):.3f}
  over {len(answered)} answered questions
- control (same sentences, scored against another question's passages):
  {sum(ctl)}/{len(ctl)} = {sum(ctl) / max(1, len(ctl)):.3f} supported. The check does reject
  text that the cited passages don't contain.

## Citation accuracy
Over the {len(ans_rows)} answerable questions that were answered:

- citations pointing at a gold page: **{n_cit_gold}/{n_cit} = {n_cit_gold / max(1, n_cit):.3f}**
  (the mock cites up to 3 passages, and only one usually holds the gold quote, so this
  precision is capped well below 1 by design)
- answers with at least one gold citation: **{n_gc}/{len(ans_rows)}
  = {n_gc / max(1, len(ans_rows)):.3f}**
- answerable questions answered at all: {len(ans_rows)}/{n_ans}

## Abstention (positive = abstain; 7 unanswerable, {n_ans} answerable)
| threshold | precision | recall | answerable wrongly abstained |
|---|---|---|---|
{row_in}
{row_cv}

Leave-one-out thresholds ranged {ts[0]:.2f} .. {ts[-1]:.2f}. With only 7 negatives, one
question moves recall by 0.14, so these are coarse estimates.
"""
    (out / "answer_quality.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
