"""Calibration and selective-prediction math. Pure functions, no I/O."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

MEASURES = ("top_prob", "margin", "entropy", "provided")


@dataclass
class Record:
    row_id: str
    qid: str
    gold: Any
    pred: Any
    correct: bool
    probabilities: dict[str, float]
    gold_key: str
    measures: dict[str, float]

    @property
    def top_prob(self) -> float:
        return self.measures["top_prob"]


def confidence_measures(probabilities: dict[str, float], provided: float | None = None) -> dict[str, float]:
    """Every way we know to collapse a distribution into one confidence number."""
    values = sorted(probabilities.values(), reverse=True)
    top = values[0]
    second = values[1] if len(values) > 1 else 0.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    max_entropy = math.log(len(values)) if len(values) > 1 else 1.0
    measures = {
        "top_prob": top,
        "margin": top - second,
        "entropy": 1.0 - entropy / max_entropy,
    }
    if provided is not None:
        measures["provided"] = float(provided)
    return measures


def wilson_lower(successes: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% Wilson lower bound on a proportion."""
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / denom)


def accuracy(records: list[Record]) -> float | None:
    return sum(r.correct for r in records) / len(records) if records else None


def reliability_bins(records: list[Record], bins: int = 10) -> list[dict]:
    """Equal-width bins over top_prob. Empty bins are omitted."""
    buckets: list[list[Record]] = [[] for _ in range(bins)]
    for record in records:
        index = min(bins - 1, int(record.top_prob * bins))
        buckets[index].append(record)
    out = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        out.append(
            {
                "lo": index / bins,
                "hi": (index + 1) / bins,
                "n": len(bucket),
                "confidence": sum(r.top_prob for r in bucket) / len(bucket),
                "accuracy": sum(r.correct for r in bucket) / len(bucket),
            }
        )
    return out


def ece(records: list[Record], bins: int = 10) -> float | None:
    """Expected calibration error: the average gap between stated and observed accuracy."""
    if not records:
        return None
    total = len(records)
    return sum(b["n"] / total * abs(b["confidence"] - b["accuracy"]) for b in reliability_bins(records, bins))


def overconfidence(records: list[Record]) -> float | None:
    """Mean stated confidence minus accuracy. Positive = overconfident."""
    if not records:
        return None
    return sum(r.top_prob for r in records) / len(records) - accuracy(records)


def brier(records: list[Record]) -> float | None:
    if not records:
        return None
    total = 0.0
    for record in records:
        for key, p in record.probabilities.items():
            y = 1.0 if key == record.gold_key else 0.0
            total += (p - y) ** 2
    return total / len(records)


def auroc(records: list[Record], measure: str) -> float | None:
    """How well the confidence measure separates correct from incorrect decisions."""
    scored = [(r.measures[measure], r.correct) for r in records if measure in r.measures]
    positives = sum(1 for _, c in scored if c)
    negatives = len(scored) - positives
    if not positives or not negatives:
        return None
    scored.sort(key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(scored):
        end = index
        while end + 1 < len(scored) and scored[end + 1][0] == scored[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(1 for k in range(index, end + 1) if scored[k][1])
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def sweep(records: list[Record], measure: str) -> list[dict]:
    """Accepted accuracy and coverage at every distinct threshold (accept when conf >= t)."""
    usable = [r for r in records if measure in r.measures]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda r: r.measures[measure], reverse=True)
    total = len(ordered)
    out = []
    correct = 0
    for index, record in enumerate(ordered):
        correct += record.correct
        is_last_of_value = index + 1 == total or ordered[index + 1].measures[measure] != record.measures[measure]
        if is_last_of_value:
            n = index + 1
            out.append(
                {
                    "threshold": record.measures[measure],
                    "n": n,
                    "coverage": n / total,
                    "accuracy": correct / n,
                    "wilson_lb": wilson_lower(correct, n),
                }
            )
    out.reverse()  # ascending threshold
    return out


def pick_threshold(
    points: list[dict], target: float, min_support: int = 30, conservative: bool = False
) -> dict | None:
    """Lowest threshold (= most coverage) whose accepted accuracy meets the target."""
    key = "wilson_lb" if conservative else "accuracy"
    for point in points:  # ascending threshold, so first hit has the most coverage
        if point["n"] >= min_support and point[key] >= target:
            return point
    return None


def at_threshold(records: list[Record], measure: str, threshold: float | None) -> dict:
    """Coverage / accepted accuracy for a fixed threshold. threshold=None means accept nothing."""
    usable = [r for r in records if measure in r.measures]
    total = len(usable)
    accepted = [] if threshold is None else [r for r in usable if r.measures[measure] >= threshold]
    correct = sum(r.correct for r in accepted)
    return {
        "n": total,
        "n_accepted": len(accepted),
        "coverage": len(accepted) / total if total else 0.0,
        "accepted_accuracy": correct / len(accepted) if accepted else None,
        "wilson_lb": wilson_lower(correct, len(accepted)) if accepted else None,
    }


def by_answer(records: list[Record], measure: str, threshold: float | None) -> dict[str, dict]:
    """Accepted accuracy per predicted answer: catches a threshold that is safe overall but bad for one class."""
    out: dict[str, dict] = {}
    if threshold is None:
        return out
    for record in records:
        if record.measures.get(measure, -1.0) < threshold:
            continue
        slot = out.setdefault(str(record.pred), {"n_accepted": 0, "correct": 0})
        slot["n_accepted"] += 1
        slot["correct"] += record.correct
    for slot in out.values():
        slot["accuracy"] = slot["correct"] / slot["n_accepted"]
    return out


def in_holdout(row_id: str, fraction: float, seed: int) -> bool:
    """Deterministic split that is stable across runs and machines."""
    if fraction <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction
