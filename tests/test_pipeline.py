import json
from importlib import resources

import pytest

from jevcal.check import check
from jevcal.cli import main
from jevcal.compile import compile_question
from jevcal.label import label_rows
from jevcal.lint import lint
from jevcal.measure import build_records, run
from jevcal.optimize import optimize
from jevcal.providers.llm import CallableLLM
from jevcal.providers.sim import SimProvider
from jevcal.runtime import Cascade
from jevcal.spec import Question, QuestionSet, load_questions, load_rows

EXAMPLE = resources.files("jevcal") / "examples" / "support"


@pytest.fixture()
def example():
    return load_questions(EXAMPLE / "questions.yaml"), load_rows(EXAMPLE / "tickets.jsonl")


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keeps .jevcal/cache out of the repo


def test_sim_is_overconfident_and_threshold_recovers_target(example):
    questions, rows = example
    predictions = run(SimProvider(seed=7), questions.questions, rows, quiet=True)
    records = build_records(questions.questions, rows, predictions)
    result = compile_question(records["department"], target=0.95)
    assert result["calibration"]["overconfidence"] > 0.03  # the simulator inflates confidence by design
    assert result["threshold"] is not None
    assert result["accuracy_all"] < 0.95  # unthresholded accuracy misses the target...
    assert result["fit"]["accepted_accuracy"] >= 0.95  # ...the threshold restores it on the fit split
    assert 0 < result["holdout"]["coverage"] < 1
    assert result["status"] in {"ok", "holdout_miss"}


def test_cli_demo_then_check_passes_and_detects_drift(tmp_path, capsys):
    assert main(["demo", "--dir", "demo"]) == 0
    lock = json.loads((tmp_path / "demo/decisions.lock.json").read_text())
    assert set(lock["questions"]) == {"is_urgent", "department", "frustration"}
    assert lock["questions"]["is_urgent"]["target"] == 0.97  # per-question target from the YAML
    assert lock["model_observed"] == ["sim-0"]
    html = (tmp_path / "demo/report.html").read_text()
    assert "<svg" in html and "simulator" in html and "Confident and wrong" in html

    assert main(["check", "--lock", "demo/decisions.lock.json", "--data", "demo/tickets.jsonl"]) == 0

    rows = load_rows(tmp_path / "demo/tickets.jsonl")
    worse = check(lock, rows, SimProvider(seed=7, skill=0.6), quiet=True)  # a much weaker "model release"
    assert not worse["ok"] and worse["failures"]


def test_cache_never_changes_results_even_with_duplicate_states(example):
    questions, rows = example
    assert len({r["state"] for r in rows}) < len(rows)  # the example really does contain duplicate texts
    fresh = run(SimProvider(seed=7), questions.questions, rows, use_cache=False, quiet=True)
    run(SimProvider(seed=7), questions.questions, rows, use_cache=True, quiet=True)  # warm the cache
    cached = run(SimProvider(seed=7), questions.questions, rows, use_cache=True, quiet=True)
    assert [p["answers"] for p in fresh] == [p["answers"] for p in cached]


def test_lint_flags_documented_weak_spots():
    questions = QuestionSet(questions={
        "neg": Question("neg", "noul", "Is the ticket not unrelated to billing and never resolved?"),
        "math": Question("math", "noul", "Does the customer mention more than 3 failed payments within the last 7 days?"),
        "ok": Question("ok", "noul", "Does the customer describe an outage that is happening now?",
                       {"true": "A service is down at the time of writing.", "false": "No current outage is described."}),
        "overlap": Question("overlap", "choice", "Which queue should take this ticket?",
                            {"billing_issue": "billing payment invoice problem", "billing_problem": "billing payment invoice issue problem",
                             "other": "anything else entirely"}),
    })
    rules = {(f.qid, f.rule) for f in lint(questions)}
    assert ("neg", "J002") in rules and ("neg", "J010") in rules
    assert ("math", "J003") in rules and ("math", "J004") in rules
    assert ("overlap", "J012") in rules
    assert not [r for q, r in rules if q == "ok"]


def test_label_fills_only_missing_labels(example):
    questions, rows = example
    rows = [dict(r, labels=dict(r["labels"])) for r in rows[:6]]
    human = rows[0]["labels"]["department"]
    for row in rows[1:]:
        row["labels"] = {}
    reply = json.dumps({"is_urgent": True, "department": "sales", "frustration": 1})
    summary = label_rows(questions, rows, CallableLLM(lambda s, u: reply, "fake:teacher"), quiet=True)
    assert summary == {"labeled": 5, "skipped": 1, "failed": []}
    assert rows[0]["labels"]["department"] == human and "labeled_by" not in rows[0]
    assert rows[3]["labels"] == {"is_urgent": True, "department": "sales", "frustration": 1}


def test_optimize_never_changes_the_answer_space(example):
    questions, rows = example
    variants = json.dumps([
        {"instructions": "Pick the team that owns the problem described in this ticket.",
         "criteria": {"billing": "money", "technical": "bugs", "sales": "buying"}},
        {"instructions": "Route it.", "criteria": {"billing": "money", "legal": "contracts"}},  # invalid: different options
    ])
    outcome = optimize(questions, rows, SimProvider(seed=7), CallableLLM(lambda s, u: variants), target=0.95,
                       only=["department"], n_variants=2, quiet=True)
    trials = outcome["report"]["department"]["trials"]
    assert [t["id"] for t in trials] == ["department", "department__v1"]
    assert outcome["questions"].questions["department"].options() == ["billing", "technical", "sales"]
    assert set(outcome["questions"].questions) == set(questions.questions)


def test_cascade_escalates_low_confidence_and_fails_closed(example, tmp_path):
    questions, rows = example
    assert main(["demo", "--dir", "demo"]) == 0
    calls = []

    def fallback(state, unsure):
        calls.append(set(unsure))
        return {qid: {"is_urgent": False, "department": "sales", "frustration": 0}[qid] for qid in unsure}

    lock_path = tmp_path / "demo/decisions.lock.json"
    cascade = Cascade.from_lock(lock_path, provider=SimProvider(seed=7), fallback=fallback, log_path=tmp_path / "log.jsonl")
    sources = set()
    for row in rows[:60]:
        for decision in cascade.decide(row["state"], row=row).values():
            sources.add(decision.source)
            if decision.source == "jev":
                assert decision.confidence >= decision.threshold and decision.answer == decision.fast_answer
    assert sources == {"jev", "fallback"} and calls
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 60

    closed = Cascade.from_lock(lock_path, provider=SimProvider(seed=7))  # no fallback configured
    unresolved = [d for row in rows[:60] for d in closed.decide(row["state"], row=row).values() if d.source == "unresolved"]
    assert unresolved and all(d.answer is None for d in unresolved)
