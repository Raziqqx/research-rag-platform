"""RRF and metric correctness on hand-computed toy data, plus randomized RRF properties."""

import math
import random

import pytest

from rag.metrics import (
    GoldSpan,
    covers,
    locate_quote,
    match_matrix,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from rag.retrieval import rrf, tokenize


# ---------------------------------------------------------------- RRF
def test_rrf_hand_computed():
    fused = dict(rrf([["a", "b", "c"], ["c", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert [d for d, _ in rrf([["a", "b", "c"], ["c", "a"]], k=60)] == ["a", "c", "b"]


def test_rrf_single_list_preserves_order():
    assert [d for d, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_tie_broken_by_best_rank_then_first_seen():
    # a and b both score 1/61 + 1/62; equal best rank 1, a seen first.
    assert [d for d, _ in rrf([["a", "b"], ["b", "a"]])] == ["a", "b"]
    # d: ranks 1,3 -> 1/61 + 1/63 = 0.032266 ; c: ranks 2,2 -> 2/62 = 0.032258.
    # 1/(k+r) is convex, so one top rank beats two middling ones by a hair.
    out = rrf([["d", "c"], ["e", "c", "d"]])
    assert [x for x, _ in out] == ["d", "c", "e"]
    assert out[0][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


@pytest.mark.parametrize("seed", range(20))
def test_rrf_properties(seed):
    rng = random.Random(seed)
    universe = list(range(30))
    lists = [rng.sample(universe, rng.randint(0, 30)) for _ in range(rng.randint(1, 4))]
    fused = rrf(lists, k=60)
    ids = [d for d, _ in fused]
    assert len(ids) == len(set(ids)) == len(set().union(*map(set, lists)))
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
    for d, s in fused:  # score equals the definition
        exp = sum(1 / (60 + lst.index(d) + 1) for lst in lists if d in lst)
        assert math.isclose(s, exp)
    # An item ranked first in every list wins.
    lists2 = [[99] + lst for lst in lists]
    assert rrf(lists2)[0][0] == 99


# ---------------------------------------------------------------- matching
def test_locate_quote_ignores_case_and_whitespace():
    text = "Intro.\n\nThe Central  Bank raised\nrates by 50 bp. End."
    s, e = locate_quote(text, "the central bank raised rates by 50 bp.")
    assert text[s:e] == "The Central  Bank raised\nrates by 50 bp."
    assert locate_quote(text, "not there") is None
    assert locate_quote(text, "  ") is None


def test_covers_threshold():
    g = GoldSpan("d", 100, 200)
    assert covers("d", 0, 150, g)  # 50 % overlap
    assert not covers("d", 0, 149, g)
    assert covers("d", 120, 130, g, min_frac=0.1)
    assert not covers("other", 100, 200, g)
    assert not covers("d", 200, 300, g)  # touching, no overlap


# ---------------------------------------------------------------- metrics
def test_recall_and_mrr_toy():
    golds = [GoldSpan("A", 0, 100), GoldSpan("B", 50, 60)]
    hits = [("C", 0, 999), ("A", 0, 80), ("B", 0, 1000), ("A", 0, 100)]
    m = match_matrix(hits, golds)
    assert m == [set(), {0}, {1}, {0}]
    assert recall_at_k(m, 2, 1) == 0.0
    assert recall_at_k(m, 2, 2) == 0.5
    assert recall_at_k(m, 2, 3) == 1.0
    assert reciprocal_rank(m) == 0.5
    assert reciprocal_rank(m, k=1) == 0.0
    assert reciprocal_rank([set(), set()]) == 0.0


def test_recall_requires_gold():
    with pytest.raises(ValueError):
        recall_at_k([set()], 0, 5)


def test_recall_counts_each_gold_once():
    assert recall_at_k([{0}, {0}, {0}], 2, 3) == 0.5


def test_mean():
    assert mean([1.0, 0.0]) == 0.5
    assert math.isnan(mean([]))


def test_tokenize():
    assert tokenize("The GDP of U.S. rose 2.5% in 2024!") == ["gdp", "u.s", "rose", "2.5", "2024"]
