"""Three chunking strategies over a document's canonical text.

Every chunker returns `Span`s (character offsets into the text), never copies, so a chunk can
always be mapped back to its pages for citation. Shared invariants (tested in
tests/test_chunking.py):
  * every span is non-empty, has no leading/trailing whitespace and is <= max_chars long;
  * spans are sorted by start;
  * recursive and semantic spans don't overlap and together cover every non-whitespace char;
  * fixed-size spans cover every char, and consecutive windows overlap by exactly `overlap`.

Sizes are in characters. ~1000 chars is ~200-250 word-piece tokens, which keeps chunks inside
all-MiniLM-L6-v2's 256-token window so the embedding sees the whole chunk.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]  # returns L2-normalised rows

STRATEGIES = ("fixed", "recursive", "semantic")
DEFAULT_MAX_CHARS = 1000
RECURSIVE_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def _trim(text: str, start: int, end: int) -> Span | None:
    """Shrink [start, end) to exclude surrounding whitespace; None if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end) if end > start else None


# --------------------------------------------------------------------------- fixed (baseline)
def fixed_size(text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = 150) -> list[Span]:
    """Sliding character windows. The naive baseline: it cuts mid-word and mid-sentence."""
    if not 0 <= overlap < max_chars:
        raise ValueError("need 0 <= overlap < max_chars")
    step = max_chars - overlap
    spans = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        spans.append(Span(start, end))
        if end == len(text):
            break
        start += step
    # Windows are taken verbatim (no trimming) so the overlap invariant stays exact; a window
    # that is pure whitespace carries no content and is dropped.
    return [s for s in spans if text[s.start : s.end].strip()]


# --------------------------------------------------------------------------- recursive
def _split_recursive(
    text: str, start: int, end: int, max_chars: int, seps: Sequence[str]
) -> list[tuple[int, int]]:
    """Split [start, end) into pieces <= max_chars, preferring the coarsest separator."""
    if end - start <= max_chars:
        return [(start, end)]
    for i, sep in enumerate(seps):
        if text.find(sep, start, end) == -1:
            continue
        pieces, pos = [], start
        while pos < end:
            hit = text.find(sep, pos, end)
            cut = end if hit == -1 else hit + len(sep)  # keep separator with left piece
            pieces.append((pos, cut))
            pos = cut
        out = []
        for a, b in pieces:
            out.extend(_split_recursive(text, a, b, max_chars, seps[i + 1 :]))
        return out
    # No separator left: hard cut.
    return [(a, min(a + max_chars, end)) for a in range(start, end, max_chars)]


def recursive(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    separators: Sequence[str] = RECURSIVE_SEPARATORS,
) -> list[Span]:
    """Split on paragraphs, then lines, sentences, clauses and words; greedily re-merge
    neighbouring pieces up to max_chars so chunks end on the most natural boundary."""
    pieces = _split_recursive(text, 0, len(text), max_chars, tuple(separators))
    merged: list[tuple[int, int]] = []
    for a, b in pieces:
        if merged and b - merged[-1][0] <= max_chars:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [s for a, b in merged if (s := _trim(text, a, b))]


# --------------------------------------------------------------------------- semantic
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n\n")


def sentence_spans(text: str) -> list[Span]:
    spans, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        if s := _trim(text, pos, m.start()):
            spans.append(s)
        pos = m.end()
    if s := _trim(text, pos, len(text)):
        spans.append(s)
    return spans


def semantic(
    text: str,
    embed: EmbedFn,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = 200,
    breakpoint_percentile: float = 90.0,
    buffer: int = 1,
) -> list[Span]:
    """Embedding-similarity breakpoints.

    Each sentence is embedded together with `buffer` neighbours on either side (single
    sentences are too short to embed reliably). A new chunk starts where the cosine distance
    between consecutive windows is above the document's `breakpoint_percentile`. Groups larger
    than max_chars are split recursively; groups smaller than min_chars merge forward.
    """
    sents = sentence_spans(text)
    if not sents:
        return []
    if len(sents) == 1:
        return recursive(text[: sents[0].end], max_chars) if len(sents[0]) > max_chars else sents

    windows = [
        " ".join(
            text[s.start : s.end]
            for s in sents[max(0, i - buffer) : min(len(sents), i + buffer + 1)]
        )
        for i in range(len(sents))
    ]
    emb = embed(windows)
    dist = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)  # rows are unit-norm
    threshold = np.percentile(dist, breakpoint_percentile)

    groups: list[tuple[int, int]] = []
    g_start = sents[0].start
    for i, d in enumerate(dist):
        if d > threshold:
            groups.append((g_start, sents[i].end))
            g_start = sents[i + 1].start
    groups.append((g_start, sents[-1].end))

    # Merge undersized groups forward while the result still fits.
    merged: list[tuple[int, int]] = []
    for a, b in groups:
        if (
            merged
            and (merged[-1][1] - merged[-1][0] < min_chars)
            and b - merged[-1][0] <= max_chars
        ):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    out: list[Span] = []
    for a, b in merged:
        if b - a <= max_chars:
            out.append(Span(a, b))
        else:
            out.extend(Span(s.start + a, s.end + a) for s in recursive(text[a:b], max_chars))
    return out


def chunk(text: str, strategy: str, embed: EmbedFn | None = None, **kw) -> list[Span]:
    if strategy == "fixed":
        return fixed_size(text, **kw)
    if strategy == "recursive":
        return recursive(text, **kw)
    if strategy == "semantic":
        if embed is None:
            raise ValueError("semantic chunking needs an embedding function")
        return semantic(text, embed, **kw)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
