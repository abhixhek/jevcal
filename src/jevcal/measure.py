"""Run a provider over a dataset and turn predictions into scored records."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .metrics import Record, confidence_measures
from .providers import Provider
from .spec import Question, SpecError, sha_of

CACHE_DIR = Path(".jevcal/cache")


def _cached_ask(provider: Provider, row: dict, questions: dict[str, Question], use_cache: bool) -> dict:
    if not (use_cache and provider.cacheable):
        return provider.ask(row["state"], questions, row=row)
    key = sha_of(
        {
            "provider": provider.cache_key(),
            "state": row["state"],
            "questions": {qid: q.to_api() for qid, q in questions.items()},
            "row": provider.row_key(row),
        }
    )
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    response = provider.ask(row["state"], questions, row=row)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response))
    return response


def run(
    provider: Provider,
    questions: dict[str, Question],
    rows: list[dict],
    *,
    concurrency: int = 8,
    use_cache: bool = True,
    quiet: bool = False,
) -> list[dict]:
    """Returns one prediction per row: {id, model, answers, usage, latency_ms} or {id, error}."""

    def work(row: dict) -> dict:
        try:
            response = _cached_ask(provider, row, questions, use_cache)
            return {"id": row["id"], **response}
        except Exception as exc:  # one bad row must not sink a 5,000-row run
            return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}

    predictions = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for done, prediction in enumerate(pool.map(work, rows), 1):
            predictions.append(prediction)
            if not quiet and (done % 50 == 0 or done == len(rows)):
                print(f"  measured {done}/{len(rows)}", file=sys.stderr)
    return predictions


def build_records(
    questions: dict[str, Question], rows: list[dict], predictions: list[dict], label_of: dict[str, str] | None = None
) -> dict[str, list[Record]]:
    """Join predictions to gold labels. `label_of` maps a question id to the label id it is scored against."""
    by_id = {p["id"]: p for p in predictions}
    records: dict[str, list[Record]] = {qid: [] for qid in questions}
    for row in rows:
        prediction = by_id.get(row["id"])
        if not prediction or "error" in prediction:
            continue
        for qid, question in questions.items():
            label_id = (label_of or {}).get(qid, qid)
            if label_id not in row["labels"] or qid not in prediction["answers"]:
                continue
            try:
                gold = question.normalize(row["labels"][label_id])
            except SpecError:
                continue
            answer = prediction["answers"][qid]
            pred = answer["answer"]
            records[qid].append(
                Record(
                    row_id=row["id"],
                    qid=qid,
                    gold=gold,
                    pred=pred,
                    correct=pred == gold,
                    probabilities=answer["probabilities"],
                    gold_key=question.key_of(gold),
                    measures=confidence_measures(answer["probabilities"], answer.get("confidence")),
                )
            )
    return records


def usage_summary(predictions: list[dict]) -> dict[str, Any]:
    good = [p for p in predictions if "error" not in p]
    tokens = [p.get("usage", {}).get("input_tokens") for p in good]
    tokens = [t for t in tokens if t]
    latencies = [p["latency_ms"] for p in good if p.get("latency_ms")]
    return {
        "n": len(predictions),
        "n_errors": len(predictions) - len(good),
        "mean_input_tokens": sum(tokens) / len(tokens) if tokens else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "models": sorted({str(p.get("model")) for p in good}),
    }
