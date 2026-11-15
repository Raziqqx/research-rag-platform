"""Answer generation with [doc_id:page] citations, validated against the retrieved passages.

Pipeline (see `generate_answer`):
  1. Abstention gate: if the best reranker score is below the threshold, answer
     NOT_FOUND without calling any provider (the corpus probably doesn't cover the question).
  2. A provider writes an answer from the passages. The prompt requires a citation after
     every claim, in the form [doc_id:page].
  3. Every citation is checked against the passages that were actually retrieved: it is
     valid only if some passage comes from that document and spans that page. Invalid
     citations are stripped from the text and reported; sentences left with no valid
     citation are flagged as unsupported.

Two providers:
  MockProvider       deterministic and extractive (no network): picks the sentence of each top
                     passage that overlaps the question most and cites it. Used by tests, CI
                     and whenever ANTHROPIC_API_KEY is unset.
  AnthropicProvider  Claude (claude-opus-5-5) via the official SDK.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag.retrieval import Hit, tokenize

NOT_FOUND = "Not found in corpus."
CITATION = re.compile(r"\[([A-Za-z0-9][\w.\-]*):(\d+)\]")
# Sentence ends at . ! ? followed by whitespace and an upper-case letter/digit/opening
# bracket, or at a paragraph break. Good enough for academic prose; abbreviations like
# "e.g. The" will over-split, which only makes extracts shorter.
_LEADING_CITES = re.compile(r"(?:\[[A-Za-z0-9][\w.\-]*:\d+\]\s*)+")
_SENT_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])|\n{2,}")

# (doc_id, absolute char offset in the document's canonical text) -> 1-based page
PageOf = Callable[[str, int], int]


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans (start, end) in `text`, whitespace trimmed, empty spans dropped."""
    spans, pos = [], 0
    for m in _SENT_BREAK.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def _page(hit: Hit, local_offset: int, page_of: PageOf | None) -> int:
    if page_of is None:
        return hit.page_start
    return min(
        max(page_of(hit.doc_id, hit.char_start + local_offset), hit.page_start), hit.page_end
    )


# --------------------------------------------------------------------------- citations
@dataclass(frozen=True)
class Citation:
    doc_id: str
    page: int


@dataclass
class CitationCheck:
    text: str  # answer with invalid citations removed
    valid: list[Citation]  # unique, in order of first appearance
    invalid: list[Citation]
    unsupported_sentences: list[str]  # sentences without any valid citation


def check_citations(answer: str, passages: Sequence[Hit]) -> CitationCheck:
    """Keep citations backed by a retrieved passage (same doc, page inside its page span)."""
    backed = {(h.doc_id, p) for h in passages for p in range(h.page_start, h.page_end + 1)}
    valid: list[Citation] = []
    invalid: list[Citation] = []

    def repl(m: re.Match) -> str:
        c = Citation(m.group(1), int(m.group(2)))
        if (c.doc_id, c.page) in backed:
            if c not in valid:
                valid.append(c)
            return m.group(0)
        if c not in invalid:
            invalid.append(c)
        return ""

    text = CITATION.sub(repl, answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    unsupported = []
    if text != NOT_FOUND:
        # A citation written after the full stop ("... rose. [d:3] Next ...") starts the next
        # fragment; move such leading citations back to the sentence they belong to.
        sents: list[str] = []
        for s, e in split_sentences(text):
            frag = text[s:e]
            lead = _LEADING_CITES.match(frag)
            if sents and lead:
                sents[-1] += " " + lead.group(0).strip()
                frag = frag[lead.end() :]
            if frag:
                sents.append(frag)
        unsupported = [x for x in sents if not CITATION.search(x) and len(tokenize(x)) > 2]
    return CitationCheck(text, valid, invalid, unsupported)


# --------------------------------------------------------------------------- providers
SYSTEM_PROMPT = (
    "You answer questions about economics and finance research papers using ONLY the numbered "
    "passages provided. Rules:\n"
    "1. Every sentence that states a fact must end with at least one citation in the exact "
    "form [doc_id:page], copied from the passage header it came from. Never cite a document "
    "or page that is not in a passage header.\n"
    "2. If the passages do not contain the answer, reply with exactly: " + NOT_FOUND + "\n"
    "3. Be concise: at most four sentences. Do not use outside knowledge."
)


def format_passages(passages: Sequence[Hit]) -> str:
    blocks = []
    for i, h in enumerate(passages, 1):
        pages = f"{h.page_start}" if h.page_start == h.page_end else f"{h.page_start}-{h.page_end}"
        cite = ", ".join(f"[{h.doc_id}:{p}]" for p in range(h.page_start, h.page_end + 1))
        blocks.append(f"Passage {i} (doc {h.doc_id}, page {pages}; cite as {cite}):\n{h.text}")
    return "\n\n".join(blocks)


class Provider(Protocol):
    name: str

    def answer(self, question: str, passages: Sequence[Hit], page_of: PageOf | None) -> str:
        """Return answer text containing [doc_id:page] citations, or NOT_FOUND."""
        ...


class MockProvider:
    """Extractive and deterministic: best-overlapping sentence from each of the top passages."""

    name = "mock-extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def answer(self, question: str, passages: Sequence[Hit], page_of: PageOf | None = None) -> str:
        q = set(tokenize(question))
        picked: list[str] = []
        seen: set[str] = set()
        for h in passages:
            best, best_score = None, 0.0
            for s, e in split_sentences(h.text):
                toks = set(tokenize(h.text[s:e]))
                # Fraction of question terms covered, with a tie-break toward shorter sentences.
                score = len(q & toks) / max(1, len(q)) - 1e-4 * (e - s) / 100
                if q & toks and score > best_score:
                    best, best_score = (s, e), score
            if best is None:
                continue
            sent = " ".join(h.text[best[0] : best[1]].split())
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"{sent} [{h.doc_id}:{_page(h, best[0], page_of)}]")
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked) if picked else NOT_FOUND


class AnthropicProvider:
    """Claude via the official SDK. Only constructed when ANTHROPIC_API_KEY is set."""

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic  # imported lazily: the mock path needs no SDK

        self.model = model
        self.client = anthropic.Anthropic()

    def answer(self, question: str, passages: Sequence[Hit], page_of: PageOf | None = None) -> str:
        prompt = f"{format_passages(passages)}\n\nQuestion: {question}"
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},  # short grounded answers; no deep reasoning needed
            # Server-side fallback on a safety-classifier refusal (routes by category).
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return NOT_FOUND
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or NOT_FOUND


def get_provider(name: str | None = None) -> Provider:
    """`name`: "mock", "anthropic" or None (= anthropic iff ANTHROPIC_API_KEY is set)."""
    name = name or os.environ.get("RAG_LLM_PROVIDER")
    if name is None:
        name = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"
    if name == "anthropic":
        return AnthropicProvider(os.environ.get("RAG_LLM_MODEL", "claude-opus-5-5"))
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider {name!r}")


# --------------------------------------------------------------------------- pipeline
@dataclass
class Answer:
    answer: str
    abstained: bool
    provider: str
    top_score: float | None
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)


def generate_answer(
    question: str,
    passages: Sequence[Hit],
    provider: Provider,
    threshold: float,
    page_of: PageOf | None = None,
    max_passages: int = 5,
) -> Answer:
    """`passages` must be reranked (Hit.score = cross-encoder score), best first."""
    top = passages[0].score if passages else None
    if top is None or top < threshold:
        return Answer(NOT_FOUND, True, provider.name, top)
    # Only passages that clear the threshold are shown to the provider.
    usable = [h for h in passages[:max_passages] if h.score >= threshold]
    raw = provider.answer(question, usable, page_of)
    if raw.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower():
        return Answer(NOT_FOUND, True, provider.name, top)
    chk = check_citations(raw, usable)
    if not chk.valid:  # nothing grounded survived validation: abstain rather than mislead
        return Answer(NOT_FOUND, True, provider.name, top, [], chk.invalid, [])
    return Answer(
        chk.text, False, provider.name, top, chk.valid, chk.invalid, chk.unsupported_sentences
    )
