"""Unit tests: sentence splitting, citation validation, mock provider, abstention gate."""

import random

import pytest

from rag.generation import (
    CITATION,
    NOT_FOUND,
    SYSTEM_PROMPT,
    MockProvider,
    check_citations,
    format_passages,
    generate_answer,
    get_provider,
    split_sentences,
)
from rag.retrieval import Hit


def hit(doc, text, p0=1, p1=None, score=5.0, start=0):
    return Hit(1, doc, p0, p1 or p0, start, start + len(text), text, score)


def test_split_sentences_spans_and_paragraphs():
    t = "First one. Second (two)! Third?\n\nNew para without stop"
    got = [t[s:e] for s, e in split_sentences(t)]
    assert got == ["First one.", "Second (two)!", "Third?", "New para without stop"]
    assert split_sentences("   ") == []


def test_split_sentences_property_spans_are_ordered_and_inside():
    rng = random.Random(0)
    alphabet = "ab. C!?\n "
    for _ in range(300):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        spans = split_sentences(t)
        prev = 0
        for s, e in spans:
            assert prev <= s < e <= len(t)
            assert not t[s].isspace() and not t[e - 1].isspace()
            prev = e


def test_valid_citation_kept_invalid_stripped_and_flagged():
    passages = [hit("doc-a", "x", 3, 4), hit("arxiv-2503.1", "y", 7)]
    ans = (
        "Spreads rose 108 bp [doc-a:4]. Growth fell sharply afterwards [doc-a:9]. "
        "Prices rose [arxiv-2503.1:7] [ghost:1]. Nothing cited here at all."
    )
    chk = check_citations(ans, passages)
    assert [(c.doc_id, c.page) for c in chk.valid] == [("doc-a", 4), ("arxiv-2503.1", 7)]
    assert [(c.doc_id, c.page) for c in chk.invalid] == [("doc-a", 9), ("ghost", 1)]
    assert "[doc-a:9]" not in chk.text and "[ghost:1]" not in chk.text
    assert "Growth fell sharply afterwards." in chk.text  # no dangling space before the full stop
    assert chk.unsupported_sentences == [
        "Growth fell sharply afterwards.",
        "Nothing cited here at all.",
    ]


def test_citation_after_full_stop_counts_for_that_sentence():
    chk = check_citations("Spreads rose by 108 bp. [d:1] Yields fell too. [d:1]", [hit("d", "x")])
    assert chk.unsupported_sentences == []


def test_every_surviving_citation_is_backed():
    rng = random.Random(1)
    for _ in range(200):
        passages = [
            hit(f"d{i}", "t", p, p + rng.randint(0, 2))
            for i, p in enumerate(rng.sample(range(1, 20), 3))
        ]
        cites = [f"[d{rng.randint(0, 4)}:{rng.randint(1, 25)}]" for _ in range(6)]
        chk = check_citations("Claim " + " ".join(cites) + ".", passages)
        for m in CITATION.finditer(chk.text):
            assert any(
                h.doc_id == m.group(1) and h.page_start <= int(m.group(2)) <= h.page_end
                for h in passages
            )
        assert len(chk.valid) + len(chk.invalid) == len(set(cites))


def test_mock_provider_is_extractive_and_cites_exact_page():
    text = "Intro text about methods.\n\nSpreads rose by 108 basis points after the reveal."
    h = hit("wb-1", text, 2, 3)

    def page_of(doc, off):  # second paragraph starts page 3
        return 3 if off >= text.index("Spreads") else 2

    out = MockProvider().answer("How much did spreads rise after the reveal?", [h], page_of)
    assert out == "Spreads rose by 108 basis points after the reveal. [wb-1:3]"
    assert MockProvider().answer("zebra quantum", [h], None) == NOT_FOUND


def test_generate_answer_abstains_below_threshold_and_on_empty():
    p = MockProvider()
    low = [hit("d", "Spreads rose a lot.", score=-3.0)]
    a = generate_answer("did spreads rise", low, p, threshold=-1.31)
    assert a.abstained and a.answer == NOT_FOUND and a.top_score == -3.0
    assert generate_answer("q", [], p, threshold=0).abstained


def test_generate_answer_uses_only_passages_above_threshold():
    ps = [
        hit("good", "Spreads rose a lot.", score=4.0),
        hit("weak", "Spreads rose a little.", score=-5.0),
    ]
    a = generate_answer("did spreads rise", ps, MockProvider(), threshold=0.0)
    assert not a.abstained
    assert [c.doc_id for c in a.citations] == ["good"]


class LyingProvider:
    name = "liar"

    def answer(self, question, passages, page_of):
        return "Made up fact [nowhere:1]."


def test_answer_with_only_invalid_citations_abstains():
    a = generate_answer("q", [hit("d", "t")], LyingProvider(), threshold=0)
    assert a.abstained and a.invalid_citations[0].doc_id == "nowhere"


def test_prompt_requires_citations_and_lists_cite_keys():
    assert "[doc_id:page]" in SYSTEM_PROMPT and NOT_FOUND in SYSTEM_PROMPT
    block = format_passages([hit("doc-a", "body", 3, 4)])
    assert "[doc-a:3], [doc-a:4]" in block and "body" in block


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RAG_LLM_PROVIDER", raising=False)
    assert get_provider().name == "mock-extractive"
    with pytest.raises(ValueError):
        get_provider("gpt")
