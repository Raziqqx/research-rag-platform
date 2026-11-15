"""Answer-quality metrics (faithfulness support, abstention thresholds) and weighted RRF."""

import importlib.util
import random
from pathlib import Path

import pytest

from rag.metrics import best_threshold, loo_abstain, precision_recall, support
from rag.retrieval import rrf

_spec = importlib.util.spec_from_file_location(
    "eval_answers", Path(__file__).resolve().parent.parent / "scripts" / "eval_answers.py"
)
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)


def test_support_exact_partial_and_empty():
    p = "Core inflation excludes food and energy prices."
    assert support("Core inflation excludes food and energy prices.", p) == 1.0
    # content tokens: core, inflation, excludes, housing -> 3/4 present
    assert support("Core inflation excludes housing.", p) == 0.75
    assert support("Bitcoin mining", p) == 0.0
    assert support("the of and", p) == 1.0  # only stopwords: nothing to support


def test_weighted_rrf_hand_computed_and_default_equals_unit_weights():
    a, b = ["x", "y"], ["y", "z"]
    fused = dict(rrf([a, b], k=60, weights=[2.0, 1.0]))
    assert fused["x"] == pytest.approx(2 / 61)
    assert fused["y"] == pytest.approx(2 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)
    assert rrf([a, b]) == rrf([a, b], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        rrf([a, b], weights=[1.0])


def test_weighted_rrf_zero_weight_reduces_to_other_list():
    rng = random.Random(1)
    a = rng.sample(range(100), 30)
    b = rng.sample(range(100), 30)
    assert [d for d, _ in rrf([a, b], weights=[0.0, 1.0])][: len(b)] == b


def test_best_threshold_separable():
    t, bacc = best_threshold([1.0, 2.0, 3.0], [-3.0, -1.0])
    assert bacc == 1.0 and -1.0 < t < 1.0 and t == 0.0


def test_loo_abstain_never_uses_own_score():
    # The lone outlier negative at 5.0 can only be caught by a threshold fitted on itself.
    scores = [1.0, 2.0, 3.0, -2.0, -3.0, 5.0]
    answerable = [True, True, True, False, False, False]
    decisions = [a for _, a in loo_abstain(scores, answerable)]
    assert decisions[:5] == [False, False, False, True, True]
    assert decisions[5] is False  # missed: honest held-out behaviour


def test_precision_recall():
    assert precision_recall([True, True, False, False], [True, False, True, False]) == (0.5, 0.5)
    p, r = precision_recall([False, False], [True, False])
    assert p != p and r == 0.0  # nan precision when nothing is predicted


def test_claims_split_sentences_and_citations():
    text = "Rates rose. [d1:2] Spreads widened by 108 bp [d2:3] [d2:4] trailing text"
    cl = eval_answers.claims(text)
    assert cl == [
        ("Rates rose.", [("d1", 2)]),
        ("Spreads widened by 108 bp", [("d2", 3), ("d2", 4)]),
        ("trailing text", []),
    ]
