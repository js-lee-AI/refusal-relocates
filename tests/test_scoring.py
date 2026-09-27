"""Scoring, calibration and data selection. None of these need torch."""

import pytest

from refusal.data import (assert_disjoint, benign_pools, fingerprint, select_by_hash,
                          shuffled_subset, split_task_pool)
from refusal.metrics import is_refusal, summarize, transition_depth
from refusal.spectrum import threshold_at_fpr


def test_refusal_and_safety_are_independent():
    responses = ["I cannot help with that request.", "A harmless explanation follows.", "I cannot " * 25]
    metrics = summarize(responses, [4., 2., 1.], ["unsafe", "safe", "safe"])
    assert metrics["coherent_refusal_rate"] == pytest.approx(2 / 3)
    assert metrics["lg3_unsafe_rate"] == pytest.approx(1 / 3)
    assert metrics["strict_refusal_rate"] == pytest.approx(1 / 3)
    assert not is_refusal("I can\u2019t help")
    assert is_refusal("I can\u2019t help", True)
    with pytest.raises(ValueError):
        summarize(["ok"], [2.], ["unknown"])


def test_threshold_respects_ties_and_budget():
    benign = [.1] * 72 + [.55] * 3
    threshold = threshold_at_fpr(benign, [.3, .54, .97], .05)
    assert threshold == .3
    assert sum(v >= threshold for v in benign) / len(benign) <= .05
    assert threshold_at_fpr([.1] * 75, [.09], .05) > .1


def test_hash_selection_preserves_order_and_rejects_missing():
    rows = ["short", "long", "hello"]
    wanted = [fingerprint(v, "prompt") for v in ["hello", "short"]]
    assert select_by_hash(iter(rows), wanted, "prompt") == ["hello", "short"]
    with pytest.raises(ValueError, match="missing"):
        select_by_hash(iter(rows[:1]), wanted, "prompt")
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([{"prompt": "hello"}], [["hello"]])


def test_benign_splits_are_disjoint():
    pairs = [{"prompt": f"prompt {i}", "response": "one two three " * (i + 1)} for i in range(100)]
    pools = benign_pools(pairs, 20)
    assert len({p["prompt"] for rows in pools.values() for p in rows}) == 80
    train, test = split_task_pool(pairs, 20)
    assert_disjoint(train, [[r["prompt"] for r in test]])
    assert shuffled_subset(pairs, 10, 42) == shuffled_subset(pairs, 10, 42)


def test_transition_threshold_conventions():
    rows = [{"layer": 5, "coherent_refusal_rate": .45}, {"layer": 8, "coherent_refusal_rate": .6}]
    assert transition_depth(rows, .8, .2) == 5
    assert transition_depth(rows, .8, .2, rescaled=True) == 8
