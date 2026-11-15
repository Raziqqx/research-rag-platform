"""Chunker invariants, checked on randomized documents (seeded, so failures reproduce)."""

import random
import zlib

import numpy as np
import pytest

from rag.chunking import fixed_size, recursive, semantic, sentence_spans
from rag.parse import PAGE_SEP, Document, clean_block, clean_page

WORDS = "rate bank inflation debt yield the of and policy market growth tax a".split()


def random_text(rng: random.Random, n_paras: int) -> str:
    paras = []
    for _ in range(n_paras):
        sents = []
        for _ in range(rng.randint(1, 8)):
            words = [rng.choice(WORDS) for _ in range(rng.randint(3, 40))]
            if rng.random() < 0.05:  # an unbreakable token longer than any chunk
                words.append("x" * rng.randint(50, 1500))
            sents.append(" ".join(words).capitalize() + ".")
        paras.append(" ".join(sents))
    return "\n\n".join(paras)


def hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic stand-in for a sentence encoder: bag of hashed words, L2-normalised."""
    out = np.zeros((len(texts), 32))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 32] += 1.0
    out += 1e-9
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def assert_common(text, spans, max_chars):
    assert spans == sorted(spans, key=lambda s: s.start)
    for s in spans:
        assert 0 < len(s) <= max_chars
        piece = text[s.start : s.end]
        assert piece.strip(), "empty chunk"


def assert_tiles_content(text, spans):
    """No overlap, and every non-whitespace character belongs to exactly one span."""
    covered = np.zeros(len(text), dtype=int)
    for s in spans:
        assert text[s.start] != " " and not text[s.end - 1].isspace()
        covered[s.start : s.end] += 1
    assert covered.max(initial=0) <= 1
    for i, ch in enumerate(text):
        if not ch.isspace():
            assert covered[i] == 1, f"char {i} not covered"


SEEDS = range(25)


@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_size_invariants(seed):
    rng = random.Random(seed)
    text = random_text(rng, rng.randint(1, 30))
    max_chars, overlap = rng.choice([(200, 0), (500, 50), (1000, 150)])
    spans = fixed_size(text, max_chars, overlap)
    assert_common(text, spans, max_chars)
    assert spans[0].start == 0 and spans[-1].end == len(text)
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.end - b.start == overlap


@pytest.mark.parametrize("seed", SEEDS)
def test_recursive_invariants(seed):
    rng = random.Random(seed)
    text = random_text(rng, rng.randint(1, 30))
    max_chars = rng.choice([150, 400, 1000])
    spans = recursive(text, max_chars)
    assert_common(text, spans, max_chars)
    assert_tiles_content(text, spans)


@pytest.mark.parametrize("seed", SEEDS)
def test_semantic_invariants(seed):
    rng = random.Random(seed)
    text = random_text(rng, rng.randint(1, 30))
    spans = semantic(text, hash_embed, max_chars=600, min_chars=100)
    assert_common(text, spans, 600)
    assert_tiles_content(text, spans)


def test_recursive_prefers_paragraph_boundaries():
    a, b = "A" * 300 + ".", "B" * 300 + "."
    text = f"{a}\n\n{b}"
    spans = recursive(text, max_chars=400)
    assert [text[s.start : s.end] for s in spans] == [a, b]


def test_semantic_breaks_at_topic_shift():
    topic1 = " ".join(["Inflation rose sharply this year."] * 6)
    topic2 = " ".join(["Penguins swim in cold oceans."] * 6)
    text = topic1 + " " + topic2
    spans = semantic(text, hash_embed, max_chars=2000, min_chars=10, breakpoint_percentile=90)
    pieces = [text[s.start : s.end] for s in spans]
    assert pieces == [topic1, topic2]


def test_fixed_size_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed_size("abc", max_chars=10, overlap=10)


def test_sentence_spans():
    text = "First one. Second (two)! Third? 4th starts with a digit.\n\nNew para"
    got = [text[s.start : s.end] for s in sentence_spans(text)]
    assert got == ["First one.", "Second (two)!", "Third?", "4th starts with a digit.", "New para"]


def test_page_mapping():
    doc = Document("d", ["page one", "", "page three"])
    assert doc.text == PAGE_SEP.join(["page one", "", "page three"])
    assert doc.page_of(0) == 1
    assert doc.page_of(len("page one") - 1) == 1
    third = doc.text.index("three")
    assert doc.page_of(third) == 3


def test_clean_block_and_page():
    assert clean_block("macro-\neconomic  policy\nworks") == "macroeconomic policy works"
    assert clean_block("rev\u00ad elations of coe\ufb03cients") == "revelations of coefficients"
    assert clean_page(["Body text.", "  12 ", "", "More."]) == "Body text.\n\nMore."
