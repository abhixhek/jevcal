"""Production wrapper: ask the fast model first, escalate what it is not sure about.

    from jevcal.runtime import Cascade, llm_fallback

    cascade = Cascade.from_lock("decisions.lock.json", fallback=llm_fallback("anthropic:claude-opus-5"))
    decisions = cascade.decide("Help! My payouts have been failing for 3 days.")
    decisions["department"].answer   # "billing"
    decisions["department"].source   # "jev" or "fallback"
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .check import questions_from_lock
from .metrics import confidence_measures
from .providers import Provider
from .spec import Question

Fallback = Callable[[Any, dict[str, Question]], dict[str, Any]]


@dataclass
class Decision:
    qid: str
    answer: Any
    source: str  # "jev" | "fallback" | "unresolved"
    confidence: float
    threshold: float | None
    fast_answer: Any
    probabilities: dict[str, float]


class Cascade:
    def __init__(self, lock: dict, provider: Provider | None = None, fallback: Fallback | None = None,
                 log_path: str | Path | None = None) -> None:
        self.lock = lock
        self.questions = questions_from_lock(lock)
        if provider is None:
            from .providers.typesafe import TypeSafeProvider

            # Pin the version the thresholds were tuned on when we know it.
            observed = lock.get("model_observed") or []
            model = observed[0] if len(observed) == 1 else lock.get("model_requested", "jev-latest")
            provider = TypeSafeProvider(model=model)
        self.provider = provider
        self.fallback = fallback
        self.log_path = Path(log_path) if log_path else None

    @classmethod
    def from_lock(cls, path: str | Path, **kwargs: Any) -> "Cascade":
        return cls(json.loads(Path(path).read_text()), **kwargs)

    def decide(self, state: Any, row: dict | None = None) -> dict[str, Decision]:
        response = self.provider.ask(state, self.questions, row=row)
        decisions: dict[str, Decision] = {}
        unsure: dict[str, Question] = {}
        for qid, question in self.questions.items():
            locked = self.lock["questions"][qid]
            answer = response["answers"][qid]
            confidence = confidence_measures(answer["probabilities"], answer.get("confidence")).get(locked["measure"], 0.0)
            threshold = locked["threshold"]
            trusted = threshold is not None and confidence >= threshold
            decisions[qid] = Decision(qid, answer["answer"] if trusted else None, "jev" if trusted else "unresolved",
                                      confidence, threshold, answer["answer"], answer["probabilities"])
            if not trusted:
                unsure[qid] = question

        if unsure and self.fallback:
            resolved = self.fallback(state, unsure)
            for qid, value in resolved.items():
                if qid in unsure:
                    decisions[qid].answer = unsure[qid].normalize(value)
                    decisions[qid].source = "fallback"

        if self.log_path:
            self._log(state, decisions)
        return decisions

    def _log(self, state: Any, decisions: dict[str, Decision]) -> None:
        # Escalations where the LLM disagrees with the fast model are the best new eval rows you can get.
        entry = {"ts": time.time(), "state": state, "decisions": {qid: asdict(d) for qid, d in decisions.items()}}
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def llm_fallback(spec: str | None = None) -> Fallback:
    from .providers.llm import LLMProvider, build_llm

    provider = LLMProvider(build_llm(spec))

    def resolve(state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        response = provider.ask(state, questions)
        return {qid: answer["answer"] for qid, answer in response["answers"].items()}

    return resolve
