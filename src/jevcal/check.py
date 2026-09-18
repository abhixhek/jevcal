"""Drift check: re-measure against a lock file and fail when the thresholds no longer hold.

`jev-latest` is an alias that moves when TypeSafe ships a release, so a threshold tuned last month
can quietly stop meaning what it meant. Run this in CI.
"""

from __future__ import annotations

from . import metrics
from .measure import build_records, run, usage_summary
from .providers import Provider
from .spec import Question


def questions_from_lock(lock: dict) -> dict[str, Question]:
    return {qid: Question.from_dict(qid, body) for qid, body in lock["questions"].items()}


def check(lock: dict, rows: list[dict], provider: Provider, *, accuracy_tolerance: float = 0.02,
          coverage_tolerance: float = 0.10, strict: bool = False, concurrency: int = 8,
          baseline_predictions: list[dict] | None = None, quiet: bool = False) -> dict:
    questions = questions_from_lock(lock)
    predictions = run(provider, questions, rows, concurrency=concurrency, use_cache=False, quiet=quiet)
    records = build_records(questions, rows, predictions)
    usage = usage_summary(predictions)

    failures, warnings, table = [], [], []
    for qid, locked in lock["questions"].items():
        now = metrics.at_threshold(records[qid], locked["measure"], locked["threshold"])
        base = locked["baseline"]
        row = {
            "qid": qid,
            "threshold": locked["threshold"],
            "target": locked["target"],
            "accepted_accuracy": now["accepted_accuracy"],
            "baseline_accepted_accuracy": base["accepted_accuracy"],
            "coverage": now["coverage"],
            "baseline_coverage": base["coverage"],
            "ece": metrics.ece(records[qid]),
            "baseline_ece": base["ece"],
        }
        table.append(row)
        if locked["threshold"] is None:
            continue
        if now["accepted_accuracy"] is None:
            failures.append(f"{qid}: nothing clears the locked threshold any more")
            continue
        if now["accepted_accuracy"] < locked["target"] - accuracy_tolerance:
            failures.append(f"{qid}: accepted accuracy {now['accepted_accuracy']:.1%} is below target {locked['target']:.0%}")
        elif base["accepted_accuracy"] is not None and now["accepted_accuracy"] < base["accepted_accuracy"] - accuracy_tolerance:
            failures.append(f"{qid}: accepted accuracy fell {base['accepted_accuracy']:.1%} -> {now['accepted_accuracy']:.1%}")
        if now["coverage"] < base["coverage"] - coverage_tolerance:
            failures.append(f"{qid}: coverage fell {base['coverage']:.1%} -> {now['coverage']:.1%} (more traffic escalates)")

    locked_models, seen_models = set(lock.get("model_observed") or []), set(usage["models"])
    if locked_models and seen_models and locked_models != seen_models:
        message = f"model changed: locked against {sorted(locked_models)}, now answered by {sorted(seen_models)}"
        (failures if strict else warnings).append(message)
    if usage["n_errors"]:
        warnings.append(f"{usage['n_errors']} of {usage['n']} rows errored during the check")

    flip_rate = None
    if baseline_predictions:
        old = {p["id"]: p for p in baseline_predictions if "error" not in p}
        pairs = flips = 0
        for prediction in predictions:
            before = old.get(prediction["id"])
            if "error" in prediction or not before:
                continue
            for qid, answer in prediction["answers"].items():
                if qid in before["answers"]:
                    pairs += 1
                    flips += answer["answer"] != before["answers"][qid]["answer"]
        flip_rate = flips / pairs if pairs else None

    return {"ok": not failures, "failures": failures, "warnings": warnings, "table": table,
            "flip_rate": flip_rate, "usage": usage}
