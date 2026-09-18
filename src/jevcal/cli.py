"""jevcal command line."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from importlib import resources
from pathlib import Path

from . import __version__, report
from .check import check as run_check
from .compile import CostModel, build_lock, compile_question, cost_summary, estimate_input_tokens, row_escalation_rate
from .label import label_rows
from .lint import lint as run_lint
from .measure import build_records, run as run_measure, usage_summary
from .optimize import optimize as run_optimize
from .providers import ProviderError, build_provider
from .providers.llm import DEFAULT_LLM, build_llm
from .spec import SpecError, load_questions, load_rows, write_jsonl


def _pct(value) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _targets(args, questions) -> dict[str, float]:
    overrides = {}
    for item in args.target_for or []:
        qid, _, value = item.partition("=")
        if qid not in questions.questions or not value:
            raise SpecError(f"--target-for expects <question>=<0..1>, got {item!r}")
        overrides[qid] = float(value)
    return {qid: overrides.get(qid) or q.target or args.target for qid, q in questions.questions.items()}


def _provider(args, questions):
    model = args.model or questions.model
    questions.model = model
    return build_provider(args.provider, model=model, llm=args.llm, seed=args.seed)


def _compile_and_write(args, questions, rows, predictions, provider_name: str) -> int:
    records = build_records(questions.questions, rows, predictions)
    usage = usage_summary(predictions)
    targets = _targets(args, questions)
    results = {}
    for qid in questions.questions:
        if not records[qid]:
            print(f"! {qid}: no labeled rows, skipped (run `jevcal label` first)", file=sys.stderr)
            continue
        results[qid] = compile_question(records[qid], target=targets[qid], measure=args.measure, holdout=args.holdout,
                                        seed=args.seed, min_support=args.min_support, conservative=args.conservative)
    if not results:
        print("nothing to compile: no question has labeled rows", file=sys.stderr)
        return 2

    cost_model = CostModel(args.jev_price, args.fallback_price_in, args.fallback_price_out,
                           args.fallback_out_tokens, args.fallback_latency_ms)
    escalation = row_escalation_rate({q: records[q] for q in results}, results, args.holdout > 0, args.holdout, args.seed)
    tokens = usage["mean_input_tokens"] or estimate_input_tokens(rows, questions)
    cost = cost_summary(escalation, tokens, usage["mean_latency_ms"], cost_model)
    lock = build_lock(questions, rows, results, usage, cost, holdout=args.holdout, seed=args.seed, provider=provider_name)

    Path(args.lock).write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    meta = {"n_rows": len(rows), "holdout": args.holdout, "provider": provider_name, "models": usage["models"],
            "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "version": __version__}
    report.write(args.report, "jevcal report", meta, questions.questions, results, cost)

    print(f"\n{'question':<22}{'threshold':>10}{'handled':>9}{'accepted acc':>14}{'all acc':>9}{'ECE':>8}  status")
    for qid, r in results.items():
        threshold = "none" if r["threshold"] is None else f"{r['threshold']:.3f}"
        print(f"{qid:<22}{threshold:>10}{_pct(r['holdout']['coverage']):>9}{_pct(r['holdout']['accepted_accuracy']):>14}"
              f"{_pct(r['accuracy_all']):>9}{_pct(r['calibration']['ece']):>8}  {r['status']}")
    print("(handled / accepted accuracy are measured on the held-out split)")
    if cost:
        print(f"\nrows that escalate: {_pct(cost['escalation_rate']).strip()}   "
              f"cascade ${cost['per_1k']['cascade']:.3f} per 1k rows vs LLM-only ${cost['per_1k']['llm_only']:.2f} "
              f"({_pct(cost['savings_vs_llm_only']).strip()} saved)")
        if len(results) > 1:
            worst = min(results, key=lambda q: results[q]["holdout"]["coverage"])
            print(f"a row escalates when any question is unsure; the bottleneck is {worst} "
                  f"(handles {_pct(results[worst]['holdout']['coverage']).strip()} at a {results[worst]['target']:.0%} target)")
    for qid, r in results.items():
        for warning in r["warnings"]:
            print(f"! {qid}: {warning}")
    if usage["n_errors"]:
        print(f"! {usage['n_errors']} rows errored and were left out")
    print(f"\nwrote {args.lock} and {args.report}")
    return 0


def cmd_lint(args) -> int:
    questions = load_questions(args.questions)
    rows = load_rows(args.data) if args.data else None
    findings = run_lint(questions, rows)
    for f in findings:
        print(f"{f.severity.upper():<5} {f.rule}  {f.qid}: {f.message}\n      fix: {f.fix}")
    counts = {s: sum(1 for f in findings if f.severity == s) for s in ("error", "warn", "info")}
    print(f"\n{len(questions.questions)} questions: {counts['error']} errors, {counts['warn']} warnings, {counts['info']} notes")
    return 1 if counts["error"] or (args.strict and counts["warn"]) else 0


def cmd_label(args) -> int:
    questions, rows = load_questions(args.questions), load_rows(args.data)
    summary = label_rows(questions, rows, build_llm(args.llm), overwrite=args.overwrite, concurrency=args.concurrency)
    write_jsonl(args.out, rows)
    print(f"labeled {summary['labeled']}, already labeled {summary['skipped']}, failed {len(summary['failed'])} -> {args.out}")
    for failure in summary["failed"][:10]:
        print(f"! row {failure['id']}: {failure['error']}")
    print("LLM labels are a teacher signal, not ground truth. Spot-check a sample before trusting thresholds built on them.")
    return 0


def cmd_measure(args) -> int:
    questions, rows = load_questions(args.questions), load_rows(args.data)
    predictions = run_measure(_provider(args, questions), questions.questions, rows,
                              concurrency=args.concurrency, use_cache=not args.no_cache)
    write_jsonl(args.out, predictions)
    usage = usage_summary(predictions)
    print(f"measured {usage['n']} rows ({usage['n_errors']} errors), answered by {', '.join(usage['models']) or 'n/a'} -> {args.out}")
    return 0


def cmd_compile(args) -> int:
    questions, rows = load_questions(args.questions), load_rows(args.data)
    predictions = [json.loads(line) for line in Path(args.preds).read_text().splitlines() if line.strip()]
    return _compile_and_write(args, questions, rows, predictions, args.provider_name)


def cmd_run(args) -> int:
    questions, rows = load_questions(args.questions), load_rows(args.data)
    predictions = run_measure(_provider(args, questions), questions.questions, rows,
                              concurrency=args.concurrency, use_cache=not args.no_cache)
    if args.preds_out:
        write_jsonl(args.preds_out, predictions)
    return _compile_and_write(args, questions, rows, predictions, args.provider)


def cmd_optimize(args) -> int:
    questions, rows = load_questions(args.questions), load_rows(args.data)
    outcome = run_optimize(questions, rows, _provider(args, questions), build_llm(args.llm), target=args.target,
                           only=args.question, n_variants=args.variants, holdout=args.holdout, seed=args.seed,
                           min_support=args.min_support, concurrency=args.concurrency, use_cache=not args.no_cache)
    for qid, entry in outcome["report"].items():
        print(f"\n{qid}: {'KEPT REWRITE ' + entry['winner'] if entry['accepted'] else 'kept original'}")
        print(f"  {'variant':<24}{'accuracy':>9}{'ECE':>8}{'held-out handled':>18}{'held-out acc':>14}")
        for t in entry["trials"]:
            print(f"  {t['id']:<24}{_pct(t['accuracy']):>9}{_pct(t['ece']):>8}{_pct(t['holdout_coverage']):>18}{_pct(t['holdout_accepted_accuracy']):>14}")
    Path(args.out).write_text(outcome["questions"].to_yaml())
    print(f"\nwrote {args.out}. A rewrite is kept only when the held-out split agrees it handles more traffic at the target.")
    return 0


def cmd_check(args) -> int:
    lock = json.loads(Path(args.lock).read_text())
    rows = load_rows(args.data)
    provider = build_provider(args.provider or lock.get("provider", "typesafe"),
                              model=args.model or lock.get("model_requested", "jev-latest"), llm=args.llm, seed=lock.get("seed", 7))
    baseline = None
    if args.baseline_preds:
        baseline = [json.loads(line) for line in Path(args.baseline_preds).read_text().splitlines() if line.strip()]
    outcome = run_check(lock, rows, provider, accuracy_tolerance=args.accuracy_tolerance,
                        coverage_tolerance=args.coverage_tolerance, strict=args.strict,
                        concurrency=args.concurrency, baseline_predictions=baseline)
    print(f"\n{'question':<22}{'threshold':>10}{'accepted acc':>14}{'was':>8}{'handled':>9}{'was':>8}")
    for row in outcome["table"]:
        threshold = "none" if row["threshold"] is None else f"{row['threshold']:.3f}"
        print(f"{row['qid']:<22}{threshold:>10}{_pct(row['accepted_accuracy']):>14}{_pct(row['baseline_accepted_accuracy']):>8}"
              f"{_pct(row['coverage']):>9}{_pct(row['baseline_coverage']):>8}")
    if outcome["flip_rate"] is not None:
        print(f"answers that changed since the baseline run: {_pct(outcome['flip_rate']).strip()}")
    for warning in outcome["warnings"]:
        print(f"! {warning}")
    for failure in outcome["failures"]:
        print(f"FAIL {failure}")
    print("\nOK: thresholds still hold" if outcome["ok"] else "\nDRIFT: recompile thresholds before trusting this model version")
    return 0 if outcome["ok"] else 1


def cmd_demo(args) -> int:
    target = Path(args.dir)
    target.mkdir(parents=True, exist_ok=True)
    source = resources.files("jevcal") / "examples" / "support"
    for name in ("questions.yaml", "tickets.jsonl"):
        with resources.as_file(source / name) as path:
            shutil.copy(path, target / name)
    print(f"copied the example into {target}/ and running it against the built-in simulator (no API key, NOT real Jev numbers)\n")
    demo_args = build_parser().parse_args([
        "run", "--questions", str(target / "questions.yaml"), "--data", str(target / "tickets.jsonl"),
        "--provider", "sim", "--target", "0.95", "--lock", str(target / "decisions.lock.json"),
        "--report", str(target / "report.html"),
    ])
    return cmd_run(demo_args)


def _add_common(parser, *, provider: bool = True) -> None:
    parser.add_argument("--questions", required=True, help="questions YAML")
    parser.add_argument("--data", required=True, help="dataset JSONL: {id, state, labels}")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    if provider:
        parser.add_argument("--provider", choices=("typesafe", "llm", "sim"), default="typesafe")
        parser.add_argument("--model", help="model id; pin a version such as jev-1.13.0 once thresholds are tuned")
        parser.add_argument("--llm", default=None, help=f"LLM spec for --provider llm (default {DEFAULT_LLM})")
        parser.add_argument("--no-cache", action="store_true", help="ignore .jevcal/cache")


def _add_compile(parser) -> None:
    parser.add_argument("--target", type=float, default=0.95, help="required accuracy among decisions the fast model keeps")
    parser.add_argument("--target-for", action="append", metavar="QUESTION=TARGET", help="per-question target, repeatable")
    parser.add_argument("--measure", default="top_prob", choices=("top_prob", "margin", "entropy", "provided", "auto"))
    parser.add_argument("--holdout", type=float, default=0.5, help="share of rows held out to verify the threshold")
    parser.add_argument("--min-support", type=int, default=30, help="minimum accepted rows behind a threshold")
    parser.add_argument("--conservative", action="store_true", help="require the 95%% lower bound, not the point estimate, to meet the target")
    parser.add_argument("--lock", default="decisions.lock.json")
    parser.add_argument("--report", default="jevcal-report.html")
    parser.add_argument("--jev-price", type=float, default=0.042, help="$ per million input tokens")
    parser.add_argument("--fallback-price-in", type=float, default=5.0, help="$ per million input tokens for the fallback LLM")
    parser.add_argument("--fallback-price-out", type=float, default=25.0)
    parser.add_argument("--fallback-out-tokens", type=int, default=60)
    parser.add_argument("--fallback-latency-ms", type=float, default=2500.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevcal", description="Calibrate, threshold, and drift-check typed decision models.")
    parser.add_argument("--version", action="version", version=f"jevcal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lint", help="check question wording against known model weak spots (no API key)")
    p.add_argument("--questions", required=True)
    p.add_argument("--data")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("label", help="fill missing gold labels with an LLM teacher")
    _add_common(p, provider=False)
    p.add_argument("--llm", default=None, help=f"default {DEFAULT_LLM}")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true", help="replace existing labels too")
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("measure", help="run the model over the dataset and save predictions")
    _add_common(p)
    p.add_argument("--out", default="predictions.jsonl")
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("compile", help="pick thresholds from saved predictions; write lock file and report")
    p.add_argument("--questions", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--provider-name", default="typesafe", help="recorded in the lock file")
    _add_compile(p)
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("run", help="measure + compile in one step")
    _add_common(p)
    _add_compile(p)
    p.add_argument("--preds-out", help="also save raw predictions here")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("optimize", help="let an LLM rewrite question wording; keep rewrites that win on held-out data")
    _add_common(p)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--question", action="append", help="only optimize this question, repeatable")
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--holdout", type=float, default=0.5)
    p.add_argument("--min-support", type=int, default=30)
    p.add_argument("--out", default="questions.optimized.yaml")
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("check", help="re-measure against a lock file; exit 1 on drift (for CI)")
    p.add_argument("--lock", default="decisions.lock.json")
    p.add_argument("--data", required=True)
    p.add_argument("--provider", choices=("typesafe", "llm", "sim"))
    p.add_argument("--model")
    p.add_argument("--llm", default=None)
    p.add_argument("--baseline-preds", help="predictions.jsonl from the run that produced the lock, to report changed answers")
    p.add_argument("--accuracy-tolerance", type=float, default=0.02)
    p.add_argument("--coverage-tolerance", type=float, default=0.10)
    p.add_argument("--strict", action="store_true", help="fail when the answering model version changed")
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("demo", help="run the bundled example against the simulator (no API key)")
    p.add_argument("--dir", default="jevcal-demo")
    p.set_defaults(fn=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (SpecError, ProviderError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
