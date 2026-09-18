"""Turn scored records into per-question thresholds, a cost estimate, and a lock file."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import __version__, metrics
from .metrics import Record
from .spec import QuestionSet, estimate_tokens, sha_of, state_text

HOLDOUT_SLACK = 0.02  # how far held-out accuracy may fall below target before we call it a miss
SMALL_SAMPLE = 100


@dataclass
class CostModel:
    jev_price_per_mtok: float = 0.042
    fallback_price_in: float = 5.0  # $/Mtok, defaults to a frontier-class LLM
    fallback_price_out: float = 25.0
    fallback_out_tokens: int = 60
    fallback_latency_ms: float = 2500.0


def pick_measure(records: list[Record], requested: str) -> str:
    available = [m for m in metrics.MEASURES if any(m in r.measures for r in records)]
    if requested != "auto":
        if requested not in available:
            raise ValueError(f"confidence measure {requested!r} is not available (have: {', '.join(available)})")
        return requested
    scored = [(metrics.auroc(records, m) or 0.0, m) for m in available]
    best = max(scored)
    # keep top_prob unless something else is clearly better: it is the only one that reads as a probability
    top = next((score for score, m in scored if m == "top_prob"), 0.0)
    return best[1] if best[0] - top > 0.01 else "top_prob"


def compile_question(
    records: list[Record],
    *,
    target: float,
    measure: str = "top_prob",
    holdout: float = 0.5,
    seed: int = 7,
    min_support: int = 30,
    conservative: bool = False,
) -> dict:
    fit = [r for r in records if not metrics.in_holdout(r.row_id, holdout, seed)]
    held = [r for r in records if metrics.in_holdout(r.row_id, holdout, seed)]
    chosen_measure = pick_measure(fit or records, measure)
    points = metrics.sweep(fit, chosen_measure)
    picked = metrics.pick_threshold(points, target, min_support=min_support, conservative=conservative)
    threshold = picked["threshold"] if picked else None

    fit_stats = metrics.at_threshold(fit, chosen_measure, threshold)
    held_stats = metrics.at_threshold(held, chosen_measure, threshold)
    full_stats = metrics.at_threshold(records, chosen_measure, threshold)

    warnings = []
    if len(records) < SMALL_SAMPLE:
        warnings.append(f"only {len(records)} labeled rows; thresholds from small samples do not hold up")
    if threshold is None:
        status = "no_threshold"
        warnings.append(f"no threshold reaches {target:.0%} with at least {min_support} accepted rows: escalate everything")
    elif held and held_stats["accepted_accuracy"] is not None and held_stats["accepted_accuracy"] < target - HOLDOUT_SLACK:
        status = "holdout_miss"
        warnings.append(
            f"held-out accepted accuracy {held_stats['accepted_accuracy']:.1%} misses the {target:.0%} target: "
            "the threshold does not generalize, add data or use --conservative"
        )
    else:
        status = "ok"

    return {
        "status": status,
        "target": target,
        "measure": chosen_measure,
        "threshold": threshold,
        "n": len(records),
        "n_fit": len(fit),
        "n_holdout": len(held),
        "accuracy_all": metrics.accuracy(records),
        "fit": fit_stats,
        "holdout": held_stats,
        "full": full_stats,
        "calibration": {
            "ece": metrics.ece(records),
            "brier": metrics.brier(records),
            "overconfidence": metrics.overconfidence(records),
            "auroc": {m: metrics.auroc(records, m) for m in metrics.MEASURES if any(m in r.measures for r in records)},
        },
        "by_answer": metrics.by_answer(records, chosen_measure, threshold),
        # confidences usually cluster near 1.0; finer bins keep the diagram readable there
        "bins": metrics.reliability_bins(records, bins=20 if min(r.top_prob for r in records) >= 0.5 else 10),
        "sweep": metrics.sweep(records, chosen_measure),
        "warnings": warnings,
    }


def row_escalation_rate(records_by_q: dict[str, list[Record]], results: dict[str, dict], holdout_only: bool,
                        holdout: float, seed: int) -> float | None:
    """A row escalates when ANY of its questions falls below that question's threshold."""
    rows: dict[str, bool] = {}
    for qid, records in records_by_q.items():
        result = results[qid]
        for record in records:
            if holdout_only and not metrics.in_holdout(record.row_id, holdout, seed):
                continue
            below = result["threshold"] is None or record.measures.get(result["measure"], -1.0) < result["threshold"]
            rows[record.row_id] = rows.get(record.row_id, False) or below
    return sum(rows.values()) / len(rows) if rows else None


def cost_summary(escalation_rate: float | None, mean_input_tokens: float | None, mean_latency_ms: float | None,
                 model: CostModel) -> dict | None:
    if escalation_rate is None or not mean_input_tokens:
        return None
    jev = mean_input_tokens * model.jev_price_per_mtok / 1e6
    fallback = (mean_input_tokens * model.fallback_price_in + model.fallback_out_tokens * model.fallback_price_out) / 1e6
    blended = jev + escalation_rate * fallback
    return {
        "escalation_rate": escalation_rate,
        "per_1k": {"jev_only": jev * 1000, "llm_only": fallback * 1000, "cascade": blended * 1000},
        "savings_vs_llm_only": 1 - blended / fallback if fallback else None,
        "latency_ms": {
            "jev": mean_latency_ms,
            "llm": model.fallback_latency_ms,
            "cascade": (mean_latency_ms or 0.0) + escalation_rate * model.fallback_latency_ms,
        },
        "assumptions": model.__dict__,
    }


def estimate_input_tokens(rows: list[dict], questions: QuestionSet) -> float:
    overhead = sum(estimate_tokens(str(q.to_api())) for q in questions.questions.values())
    return sum(estimate_tokens(state_text(r["state"])) for r in rows) / len(rows) + overhead


def build_lock(questions: QuestionSet, rows: list[dict], results: dict[str, dict], usage: dict, cost: dict | None,
               *, holdout: float, seed: int, provider: str) -> dict:
    return {
        "jevcal_version": __version__,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "provider": provider,
        "model_requested": questions.model,
        "model_observed": usage.get("models", []),
        "questions_sha": questions.sha(),
        "dataset_sha": sha_of([[r["id"], r["state"], r["labels"]] for r in rows]),
        "n_rows": len(rows),
        "holdout": holdout,
        "seed": seed,
        "questions": {
            qid: {
                **questions.questions[qid].to_api(),
                "target": result["target"],
                "measure": result["measure"],
                "threshold": result["threshold"],
                "status": result["status"],
                "baseline": {
                    "accuracy_all": result["accuracy_all"],
                    "accepted_accuracy": result["full"]["accepted_accuracy"],
                    "coverage": result["full"]["coverage"],
                    "holdout_accepted_accuracy": result["holdout"]["accepted_accuracy"],
                    "holdout_coverage": result["holdout"]["coverage"],
                    "ece": result["calibration"]["ece"],
                },
            }
            for qid, result in results.items()
        },
        "cost": cost,
    }
