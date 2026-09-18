"""A simulated decision model for demos and tests. It is NOT Jev.

It reads the gold label, is right with a hidden probability, and reports a confidence that is
deliberately higher than that hidden probability (overconfident). That gives jevcal something
realistic to find: decent accuracy, inflated confidence, and a threshold that actually matters.
Rewording a question changes the noise, so `jevcal optimize` has something to chew on too.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ..spec import Question, estimate_tokens, state_text
from . import Provider, ProviderError


class SimProvider(Provider):
    name = "sim"

    def __init__(self, seed: int = 7, skill: float = 12.0, overconfidence: float = 1.8) -> None:
        self.seed = seed
        self.skill = skill
        self.overconfidence = overconfidence

    def cache_key(self) -> dict:
        return {"provider": self.name, "seed": self.seed, "skill": self.skill, "over": self.overconfidence}

    def row_key(self, row: dict):
        return [row["id"], row.get("labels")]  # the simulator reads both

    def ask(self, state: Any, questions: dict[str, Question], row: dict | None = None) -> dict:
        if row is None:
            raise ProviderError("the sim provider needs the dataset row (it reads gold labels)")
        answers = {}
        for qid, question in questions.items():
            label_id = qid.split("__")[0]  # optimize evaluates variants as <qid>__v<n>
            if label_id not in row.get("labels", {}):
                raise ProviderError(f"sim provider: row {row['id']} has no gold label for {label_id!r}")
            gold = question.normalize(row["labels"][label_id])
            answers[qid] = self._answer(question, gold, row["id"])
        text = state_text(state)
        return {
            "model": "sim-0",
            "answers": answers,
            "usage": {"input_tokens": estimate_tokens(text) + 40 * len(questions), "output_tokens": 0},
            "latency_ms": 110.0 + 4.0 * len(questions),
        }

    def _answer(self, question: Question, gold: Any, row_id: str) -> dict:
        wording = hashlib.sha256(question.instructions.encode()).hexdigest()[:8]
        digest = hashlib.sha256(f"{self.seed}|{row_id}|{question.id.split('__')[0]}|{wording}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        options = question.options()
        chance = 1.0 / len(options)

        hidden = chance + (1.0 - chance) * rng.random() ** (1.0 / self.skill)
        is_correct = rng.random() < hidden
        reported = max(hidden ** (1.0 / self.overconfidence), chance + 0.02)

        gold_key = question.key_of(gold)
        top_key = gold_key if is_correct else rng.choice([o for o in options if o != gold_key])
        rest = (1.0 - reported) / (len(options) - 1)
        probabilities = {key: reported if key == top_key else rest for key in options}

        if question.type == "noul":
            return {"type": "noul", "answer": top_key == "true", "probabilities": probabilities, "confidence": None}
        if question.type == "choice":
            return {"type": "choice", "answer": top_key, "probabilities": probabilities, "confidence": None}
        expected = sum(int(key) * p for key, p in probabilities.items())
        return {"type": "score", "answer": int(top_key), "score": expected, "probabilities": probabilities, "confidence": None}
