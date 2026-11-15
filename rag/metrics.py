"""Retrieval metrics and gold-passage matching.

A gold passage is (doc_id, page, quote). The quote is located in the document's canonical text
to get a character range; a retrieved chunk "covers" a gold passage when it is from the same
document and overlaps at least `min_frac` of the quote's characters. Matching on character
ranges (not chunk ids) is what makes the same eval set usable across chunking strategies.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    start: int
    end: int


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find `quote` in `text`, ignoring case and whitespace differences.

    Returns (start, end) offsets in the original text, or None.
    """
    norm_chars, index = [], []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                norm_chars.append(" ")
                index.append(i)
            prev_space = True
        else:
            norm_chars.append(ch.lower())
            index.append(i)
            prev_space = False
    norm = "".join(norm_chars)
    q = _WS.sub(" ", quote.strip().lower())
    pos = norm.find(q)
    if pos == -1 or not q:
        return None
    return index[pos], index[pos + len(q) - 1] + 1


def covers(doc_id: str, start: int, end: int, gold: GoldSpan, min_frac: float = 0.5) -> bool:
    if doc_id != gold.doc_id:
        return False
    overlap = min(end, gold.end) - max(start, gold.start)
    return overlap > 0 and overlap >= min_frac * (gold.end - gold.start)


def match_matrix(
    hits: Sequence[tuple[str, int, int]], golds: Sequence[GoldSpan], min_frac: float = 0.5
) -> list[set[int]]:
    """For each retrieved (doc_id, start, end), the indices of the gold spans it covers."""
    return [
        {g for g, gold in enumerate(golds) if covers(d, s, e, gold, min_frac)} for d, s, e in hits
    ]


def recall_at_k(matches: Sequence[set[int]], n_gold: int, k: int) -> float:
    """Fraction of gold passages covered by at least one of the top-k results."""
    if n_gold == 0:
        raise ValueError("recall is undefined for a question with no gold passages")
    found = set().union(*matches[:k]) if matches[:k] else set()
    return len(found) / n_gold


def reciprocal_rank(matches: Sequence[set[int]], k: int | None = None) -> float:
    """1 / rank of the first result covering any gold passage (0 if none within k)."""
    for rank, m in enumerate(matches[:k] if k else matches, start=1):
        if m:
            return 1.0 / rank
    return 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")
