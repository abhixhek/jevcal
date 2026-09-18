"""Let an LLM rewrite question wording, then keep a rewrite only if held-out data says it is better.

All variants of a question are sent in ONE request per row: the decision model evaluates questions
in parallel, so testing five wordings costs about the same as testing one.
"""

from __future__ import annotations

import json

from . import metrics
from .compile import compile_question
from .measure import build_records, run
from .providers import Provider
from .providers.llm import LLM, extract_json
from .spec import Question, QuestionSet, SpecError, state_text

REWRITE_SYSTEM = """You improve questions for a decision model (TypeSafe Jev). It does not generate text: it \
reads a state and returns a probability distribution over predefined answers. Known weaknesses:
- It answers the question literally and does not infer intent. Spell out scope and boundary cases.
- Negations and double negatives hurt accuracy. Phrase positively.
- It cannot count, do arithmetic, or compare dates or numbers.
- Multi-hop or compound questions hurt accuracy. One judgment per question.
- Vague words need definitions in the criteria.
Rewrite the question so the same labels stay correct. You MUST keep the question type, and for choice \
questions the exact same option keys, and for score questions the same number of levels in the same order. \
Reply with a JSON array of variants, each {"instructions": str, "criteria": ...}, and nothing else."""


def propose_variants(llm: LLM, question: Question, errors: list[dict], n: int) -> list[Question]:
    user = (
        f"Question to improve:\n{json.dumps(question.to_api(), indent=2, ensure_ascii=False)}\n\n"
        f"Cases the current wording gets wrong (state, correct answer, model answer, model confidence):\n"
        f"{json.dumps(errors, indent=2, ensure_ascii=False)}\n\n"
        f"Write {n} meaningfully different variants."
    )
    parsed = extract_json(llm.complete(REWRITE_SYSTEM, user))
    if not isinstance(parsed, list):
        return []
    variants = []
    for index, body in enumerate(parsed[:n], 1):
        try:
            variant = Question(
                id=f"{question.id}__v{index}",
                type=question.type,
                instructions=str(body["instructions"]),
                criteria=body.get("criteria", question.criteria),
                target=question.target,
            )
        except (SpecError, KeyError, TypeError):
            continue
        if variant.options() != question.options():
            continue  # changed the answer space: labels would no longer line up
        variants.append(variant)
    return variants


def _errors(question: Question, records: list[metrics.Record], rows_by_id: dict[str, dict], limit: int = 8) -> list[dict]:
    wrong = sorted((r for r in records if not r.correct), key=lambda r: -r.top_prob)[:limit]
    return [
        {
            "state": state_text(rows_by_id[r.row_id]["state"])[:600],
            "correct_answer": r.gold,
            "model_answer": r.pred,
            "model_confidence": round(r.top_prob, 3),
        }
        for r in wrong
    ]


def _score(result: dict) -> tuple:
    # more coverage at the target wins; then accuracy; then better calibration
    return (result["fit"]["coverage"], result["accuracy_all"] or 0.0, -(result["calibration"]["ece"] or 1.0))


def optimize(questions: QuestionSet, rows: list[dict], provider: Provider, llm: LLM, *, target: float,
             only: list[str] | None = None, n_variants: int = 4, holdout: float = 0.5, seed: int = 7,
             min_support: int = 30, concurrency: int = 8, use_cache: bool = True, quiet: bool = False) -> dict:
    rows_by_id = {r["id"]: r for r in rows}
    selected = {qid: q for qid, q in questions.questions.items() if not only or qid in only}

    baseline_preds = run(provider, selected, rows, concurrency=concurrency, use_cache=use_cache, quiet=quiet)
    baseline_records = build_records(selected, rows, baseline_preds)

    candidates: dict[str, Question] = {}
    label_of: dict[str, str] = {}
    for qid, question in selected.items():
        fit = [r for r in baseline_records[qid] if not metrics.in_holdout(r.row_id, holdout, seed)]
        for variant in propose_variants(llm, question, _errors(question, fit, rows_by_id), n_variants):
            candidates[variant.id] = variant
            label_of[variant.id] = qid

    variant_records: dict[str, list[metrics.Record]] = {}
    if candidates:
        variant_preds = run(provider, candidates, rows, concurrency=concurrency, use_cache=use_cache, quiet=quiet)
        variant_records = build_records(candidates, rows, variant_preds, label_of=label_of)

    report: dict[str, dict] = {}
    improved = dict(questions.questions)
    for qid, question in selected.items():
        q_target = question.target or target
        kwargs = dict(target=q_target, holdout=holdout, seed=seed, min_support=min_support)
        original = compile_question(baseline_records[qid], **kwargs)
        trials = [{"id": qid, "question": question, "result": original}]
        for vid, variant in candidates.items():
            if label_of[vid] == qid and variant_records.get(vid):
                trials.append({"id": vid, "question": variant, "result": compile_question(variant_records[vid], **kwargs)})

        best = max(trials, key=lambda t: _score(t["result"]))
        accepted = best["id"] != qid and _beats_on_holdout(best["result"], original, q_target)
        if accepted:
            winner = best["question"]
            improved[qid] = Question(id=qid, type=winner.type, instructions=winner.instructions,
                                     criteria=winner.criteria, target=question.target)
        report[qid] = {
            "accepted": accepted,
            "winner": best["id"] if accepted else qid,
            "trials": [
                {
                    "id": t["id"],
                    "instructions": t["question"].instructions,
                    "accuracy": t["result"]["accuracy_all"],
                    "ece": t["result"]["calibration"]["ece"],
                    "fit_coverage": t["result"]["fit"]["coverage"],
                    "holdout_coverage": t["result"]["holdout"]["coverage"],
                    "holdout_accepted_accuracy": t["result"]["holdout"]["accepted_accuracy"],
                }
                for t in trials
            ],
        }
    return {"questions": QuestionSet(questions=improved, model=questions.model), "report": report}


def _beats_on_holdout(candidate: dict, original: dict, target: float) -> bool:
    """The fit split picked the winner; the held-out split has to agree before we keep it."""
    held = candidate["holdout"]
    if candidate["threshold"] is None:
        return False
    if held["accepted_accuracy"] is not None and held["accepted_accuracy"] < target - 0.02:
        return False
    gained_coverage = held["coverage"] - original["holdout"]["coverage"]
    return gained_coverage > 0.02 or (original["threshold"] is None and held["coverage"] > 0)
