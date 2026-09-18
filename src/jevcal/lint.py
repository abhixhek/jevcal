"""Static checks for question wording, based on TypeSafe's published Jev 1.13 weak spots.

Source: https://docs.typesafe.ai/model-jaggedness/jev-1.13
No API key needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .spec import MAX_OPTIONS, Question, QuestionSet, estimate_tokens, state_text

SEVERITIES = ("error", "warn", "info")

NEGATION = re.compile(r"\b(not|never|no longer|isn't|aren't|doesn't|don't|didn't|won't|cannot|can't|without|except|unless|neither|nor)\b", re.I)
ARITHMETIC = re.compile(r"\b(how many|count of|number of|at least \d+|at most \d+|more than \d+|fewer than \d+|less than \d+|sum of|total of|average|percent(age)?)\b", re.I)
DATETIME = re.compile(r"\b(within the (last|past|next)|older than|newer than|earlier than|later than|expired?|overdue|past due|(before|after|since|until) \d|\d+\s*(minutes?|hours?|days?|weeks?|months?|years?))\b", re.I)
NUMERIC = re.compile(r"(\b(greater than|less than|exceeds?|at least|at most|above|below|over|under)\s+[$€£]?\d|[<>]=?\s*[$€£]?\d)", re.I)
MULTI_HOP = re.compile(r"\b(whose|of the \w+ of|which of .+ that|if .+ then .+ otherwise)\b", re.I)
COMPOUND = re.compile(r"\b(and|or|as well as)\b", re.I)
VAGUE = re.compile(r"\b(good|bad|appropriate|relevant|quality|suitable|acceptable|reasonable|important|interesting)\b", re.I)


@dataclass
class Finding:
    rule: str
    severity: str
    qid: str
    message: str
    fix: str


def _text_of(question: Question) -> str:
    parts = [question.instructions]
    if isinstance(question.criteria, dict):
        parts += [str(v) for v in question.criteria.values() if v]
    elif isinstance(question.criteria, list):
        parts += [str(v) for v in question.criteria]
    return " \n ".join(parts)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", text.lower())}


def lint_question(question: Question) -> list[Finding]:
    qid = question.id
    text = _text_of(question)
    findings: list[Finding] = []

    def add(rule: str, severity: str, message: str, fix: str) -> None:
        findings.append(Finding(rule, severity, qid, message, fix))

    negations = NEGATION.findall(text)
    if len(negations) >= 2:
        add("J002", "error", f"multiple negations ({', '.join(sorted(set(n.lower() for n in negations)))})",
            "Jev reads negations literally and double negatives cut accuracy. Ask the positive version and flip the answer in code.")
    elif negations:
        add("J001", "warn", f"negation ({negations[0].lower()!r})",
            "P(noul) and 1 - P(negated noul) are not interchangeable on Jev. Prefer the positive phrasing.")

    if ARITHMETIC.search(text):
        add("J003", "warn", "asks the model to count or do arithmetic",
            "Jev is not a calculator. Compute the number in code and put the result in the state.")
    if DATETIME.search(text):
        add("J004", "warn", "depends on a date or duration comparison",
            "Dates are read as text. Precompute the comparison (e.g. days_overdue: 12) and ask about that field.")
    if NUMERIC.search(text):
        add("J005", "warn", "depends on a numeric comparison",
            "Do the comparison in code, or describe the bands in words inside the criteria.")
    if MULTI_HOP.search(text):
        add("J006", "info", "looks like a multi-hop question",
            "Indirection lowers accuracy. Split into atomic questions and combine the answers in code; extra questions are nearly free.")

    words = question.instructions.split()
    if len(words) < 4:
        add("J007", "warn", "instructions are very short",
            "Jev answers the question as written and will not infer intent. Say exactly what should count.")
    if VAGUE.search(question.instructions) and not question.criteria:
        add("J008", "info", "subjective wording with no criteria",
            "Define what the subjective word means in `criteria`, including the boundary cases.")

    if question.type == "noul":
        if COMPOUND.search(question.instructions):
            add("J010", "warn", "compound yes/no question (and / or)",
                "Ask one thing per noul. Several nouls in one request cost almost nothing extra.")
        if not question.criteria:
            add("J009", "info", "noul has no criteria", "Describe what a yes and a no mean, especially near the boundary.")
        elif str(question.criteria.get("true", "")).strip().lower() == str(question.criteria.get("false", "")).strip().lower():
            add("J014", "error", "true and false criteria are identical", "Contradictory guidance confuses the model. Make them distinct.")

    if question.type == "choice":
        options = question.criteria
        if len(options) > MAX_OPTIONS:
            add("J011", "error", f"{len(options)} options exceeds the {MAX_OPTIONS}-option limit", "Use hierarchical classification.")
        undescribed = [k for k, v in options.items() if not v]
        if len(undescribed) > len(options) / 2:
            add("J009", "info", f"{len(undescribed)} of {len(options)} options have no description",
                "Option descriptions are where domain rules live. Describe each option.")
        keys = list(options)
        for i, first in enumerate(keys):
            for second in keys[i + 1:]:
                a, b = _tokens(f"{first} {options[first] or ''}"), _tokens(f"{second} {options[second] or ''}")
                if a and b and len(a & b) / len(a | b) > 0.6:
                    add("J012", "warn", f"options {first!r} and {second!r} overlap heavily",
                        "Overlapping options flatten the distribution and tank confidence. Merge them or sharpen the boundary.")

    if question.type == "score" and len(question.criteria) > 7:
        add("J013", "warn", f"{len(question.criteria)} score levels",
            "Fine-grained scales produce low confidence. Use 3 to 5 levels with clear descriptions.")

    return findings


def lint_data(questions: QuestionSet, rows: list[dict]) -> list[Finding]:
    findings = []
    sizes = [estimate_tokens(state_text(row["state"])) for row in rows]
    mean, largest = sum(sizes) / len(sizes), max(sizes)
    if largest > 32_000:
        findings.append(Finding("J020", "error", "*", f"largest state is ~{largest:,} tokens, above the 32k state limit",
                                "Trim or chunk the state before sending it."))
    elif mean > 8_000:
        findings.append(Finding("J021", "warn", "*", f"states average ~{mean:,.0f} tokens",
                                "Irrelevant state is a distractor and accuracy falls as it grows. Send only what the question needs."))
    unlabeled = {qid: sum(1 for r in rows if qid not in r["labels"]) for qid in questions.questions}
    for qid, missing in unlabeled.items():
        if missing:
            findings.append(Finding("J022", "info", qid, f"{missing} of {len(rows)} rows have no label",
                                    "Run `jevcal label` to fill them with an LLM teacher."))
    return findings


def lint(questions: QuestionSet, rows: list[dict] | None = None) -> list[Finding]:
    findings = [f for q in questions.questions.values() for f in lint_question(q)]
    if rows:
        findings += lint_data(questions, rows)
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    return findings
