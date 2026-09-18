"""Questions, datasets, and the canonical answer representation.

Canonical answers: noul -> bool, choice -> str (option key), score -> int (level index).
Probability maps always use string keys: "true"/"false", option keys, or "0".."n-1".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

QUESTION_TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


class SpecError(ValueError):
    pass


def _str_key(key: Any) -> str:
    # YAML parses bare `true:` / `false:` keys as booleans.
    if isinstance(key, bool):
        return "true" if key else "false"
    return str(key)


@dataclass
class Question:
    id: str
    type: str
    instructions: str
    criteria: Any = None
    target: float | None = None  # optional per-question accuracy target

    def __post_init__(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise SpecError(f"{self.id}: unknown question type {self.type!r}")
        if self.type == "noul" and isinstance(self.criteria, dict):
            self.criteria = {_str_key(k): v for k, v in self.criteria.items()}
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise SpecError(f"{self.id}: choice needs a criteria map with at least 2 options")
            self.criteria = {_str_key(k): v for k, v in self.criteria.items()}
        if self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise SpecError(f"{self.id}: score needs a criteria list with at least 2 levels")

    def options(self) -> list[str]:
        """Probability-map keys, in declaration order."""
        if self.type == "noul":
            return ["true", "false"]
        if self.type == "choice":
            return list(self.criteria)
        return [str(i) for i in range(len(self.criteria))]

    def key_of(self, answer: Any) -> str:
        """Canonical answer -> probability-map key."""
        if self.type == "noul":
            return "true" if answer else "false"
        return str(answer)

    def normalize(self, value: Any) -> Any:
        """Coerce a label / LLM output into the canonical answer, or raise SpecError."""
        if self.type == "noul":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
        elif self.type == "choice":
            text = str(value).strip()
            if text in self.criteria:
                return text
            lowered = {k.lower(): k for k in self.criteria}
            if text.lower() in lowered:
                return lowered[text.lower()]
        else:
            if isinstance(value, bool):
                raise SpecError(f"{self.id}: {value!r} is not a score level")
            if isinstance(value, (int, float)) and float(value).is_integer():
                index = int(value)
                if 0 <= index < len(self.criteria):
                    return index
            text = str(value).strip()
            if text.isdigit() and int(text) < len(self.criteria):
                return int(text)
            for index, level in enumerate(self.criteria):
                if text.lower() == str(level).strip().lower():
                    return index
        raise SpecError(f"{self.id}: {value!r} is not a valid {self.type} answer")

    def to_api(self) -> dict:
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body

    def to_dict(self) -> dict:
        body = self.to_api()
        if self.target is not None:
            body["target"] = self.target
        return body

    @classmethod
    def from_dict(cls, qid: str, body: dict) -> "Question":
        if not isinstance(body, dict) or "type" not in body or "instructions" not in body:
            raise SpecError(f"{qid}: a question needs `type` and `instructions`")
        return cls(
            id=qid,
            type=body["type"],
            instructions=str(body["instructions"]),
            criteria=body.get("criteria"),
            target=body.get("target"),
        )


@dataclass
class QuestionSet:
    questions: dict[str, Question]
    model: str = "jev-latest"
    source: str | None = None
    extra: dict = field(default_factory=dict)

    def sha(self) -> str:
        payload = {qid: q.to_api() for qid, q in self.questions.items()}
        return sha_of(payload)

    def to_yaml(self) -> str:
        doc = {"model": self.model, "questions": {qid: q.to_dict() for qid, q in self.questions.items()}}
        return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


def load_questions(path: str | Path) -> QuestionSet:
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("questions"), dict):
        raise SpecError(f"{path}: expected a top-level `questions:` map")
    questions = {str(qid): Question.from_dict(str(qid), body) for qid, body in doc["questions"].items()}
    if not questions:
        raise SpecError(f"{path}: no questions defined")
    return QuestionSet(questions=questions, model=str(doc.get("model", "jev-latest")), source=str(path))


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path}:{number}: invalid JSON ({exc})") from exc
        if "state" not in row:
            raise SpecError(f"{path}:{number}: row has no `state`")
        row.setdefault("id", str(number))
        row["id"] = str(row["id"])
        if row["id"] in seen:
            raise SpecError(f"{path}:{number}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        row.setdefault("labels", {})
        rows.append(row)
    if not rows:
        raise SpecError(f"{path}: dataset is empty")
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha_of(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
