import math

from jevcal import metrics
from jevcal.metrics import Record, confidence_measures


def rec(row_id, conf, correct, qid="q"):
    probabilities = {"true": conf, "false": 1 - conf}
    return Record(row_id=str(row_id), qid=qid, gold=True, pred=correct, correct=correct,
                  probabilities=probabilities, gold_key="true", measures=confidence_measures(probabilities))


def test_confidence_measures_binary():
    m = confidence_measures({"true": 0.9, "false": 0.1})
    assert m["top_prob"] == 0.9
    assert math.isclose(m["margin"], 0.8)
    assert 0 < m["entropy"] < 1
    assert "provided" not in m
    assert confidence_measures({"a": 0.5, "b": 0.5}, provided=0.2)["provided"] == 0.2


def test_uniform_distribution_has_zero_entropy_confidence():
    assert math.isclose(confidence_measures({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})["entropy"], 0.0, abs_tol=1e-9)


def test_ece_zero_when_perfectly_calibrated():
    # 10 decisions at 0.8 confidence, 8 correct
    records = [rec(i, 0.8, i < 8) for i in range(10)]
    assert math.isclose(metrics.ece(records), 0.0, abs_tol=1e-9)
    assert math.isclose(metrics.overconfidence(records), 0.0, abs_tol=1e-9)


def test_ece_detects_overconfidence():
    records = [rec(i, 0.95, i < 6) for i in range(10)]  # says 95%, is 60%
    assert math.isclose(metrics.ece(records), 0.35, abs_tol=1e-9)
    assert metrics.overconfidence(records) > 0.3


def test_auroc_perfect_and_random():
    perfect = [rec(i, 0.9, True) for i in range(5)] + [rec(i + 5, 0.6, False) for i in range(5)]
    assert metrics.auroc(perfect, "top_prob") == 1.0
    tied = [rec(i, 0.7, i % 2 == 0) for i in range(10)]
    assert metrics.auroc(tied, "top_prob") == 0.5
    assert metrics.auroc([rec(1, 0.9, True)], "top_prob") is None


def test_sweep_and_pick_threshold():
    # high-confidence decisions are all right, low-confidence ones are all wrong
    records = [rec(i, 0.9, True) for i in range(40)] + [rec(i + 40, 0.6, False) for i in range(40)]
    points = metrics.sweep(records, "top_prob")
    assert [p["threshold"] for p in points] == [0.6, 0.9]
    assert points[0]["coverage"] == 1.0 and points[0]["accuracy"] == 0.5
    assert points[1]["coverage"] == 0.5 and points[1]["accuracy"] == 1.0
    picked = metrics.pick_threshold(points, target=0.95, min_support=30)
    assert picked["threshold"] == 0.9


def test_pick_threshold_prefers_most_coverage():
    records = [rec(i, 0.7, True) for i in range(50)] + [rec(i + 50, 0.9, True) for i in range(50)]
    picked = metrics.pick_threshold(metrics.sweep(records, "top_prob"), target=0.95, min_support=30)
    assert picked["threshold"] == 0.7 and picked["coverage"] == 1.0


def test_pick_threshold_respects_min_support_and_conservative():
    records = [rec(i, 0.99, True) for i in range(10)] + [rec(i + 10, 0.6, False) for i in range(90)]
    points = metrics.sweep(records, "top_prob")
    assert metrics.pick_threshold(points, 0.95, min_support=30) is None
    assert metrics.pick_threshold(points, 0.95, min_support=5)["threshold"] == 0.99
    # 10 of 10 correct is not enough evidence for 95% at a 95% lower bound
    assert metrics.pick_threshold(points, 0.95, min_support=5, conservative=True) is None


def test_wilson_lower_bound():
    assert metrics.wilson_lower(0, 0) == 0.0
    assert 0.75 < metrics.wilson_lower(10, 10) < 0.8
    assert metrics.wilson_lower(990, 1000) > 0.98


def test_at_threshold_none_accepts_nothing():
    stats = metrics.at_threshold([rec(1, 0.9, True)], "top_prob", None)
    assert stats["coverage"] == 0.0 and stats["accepted_accuracy"] is None


def test_holdout_split_is_deterministic_and_roughly_balanced():
    ids = [str(i) for i in range(2000)]
    first = [metrics.in_holdout(i, 0.5, 7) for i in ids]
    assert first == [metrics.in_holdout(i, 0.5, 7) for i in ids]
    assert 0.45 < sum(first) / len(first) < 0.55
    assert not any(metrics.in_holdout(i, 0.0, 7) for i in ids)
