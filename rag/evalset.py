"""Load the eval set (eval/questions.jsonl) and resolve gold quotes to character spans.

Resolution doubles as validation: every quote must be found in its document's canonical text
and must start on the page the annotator recorded, otherwise loading fails loudly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag.config import REPO_ROOT
from rag.metrics import GoldSpan, locate_quote
from rag.parse import Document

EVAL_PATH = REPO_ROOT / "eval" / "questions.jsonl"


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    answerable: bool
    gold: list[GoldSpan]
    answer: str | None


def load_questions(get_doc: Callable[[str], Document], path: Path = EVAL_PATH) -> list[Question]:
    out, errors = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        spans = []
        for g in raw["gold"]:
            doc = get_doc(g["doc_id"])
            loc = locate_quote(doc.text, g["quote"])
            if loc is None:
                errors.append(f"{raw['id']}: quote not found in {g['doc_id']}: {g['quote'][:60]!r}")
                continue
            if doc.page_of(loc[0]) != g["page"]:
                errors.append(
                    f"{raw['id']}: quote is on page {doc.page_of(loc[0])}, not {g['page']}"
                )
            spans.append(GoldSpan(g["doc_id"], *loc))
        if raw["answerable"] != bool(raw["gold"]):
            errors.append(f"{raw['id']}: answerable flag disagrees with gold list")
        out.append(
            Question(
                raw["id"], raw["type"], raw["question"], raw["answerable"], spans, raw.get("answer")
            )
        )
    if errors:
        raise ValueError("invalid eval set:\n  " + "\n  ".join(errors))
    return out


def doc_loader(conn) -> Callable[[str], Document]:
    """Cached doc_id -> Document rebuilt from stored pages, so offsets match ingest exactly."""
    from functools import cache

    @cache
    def get_doc(doc_id: str) -> Document:
        pages = [
            r[0]
            for r in conn.execute(
                "SELECT text FROM pages WHERE doc_id=%s ORDER BY page_no", (doc_id,)
            )
        ]
        if not pages:
            raise LookupError(f"document {doc_id} is not ingested")
        return Document(doc_id, pages)

    return get_doc
