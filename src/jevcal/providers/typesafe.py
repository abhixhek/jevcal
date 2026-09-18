"""TypeSafe System One API (Jev). Plain HTTPS, no SDK dependency.

Docs: https://docs.typesafe.ai/api
Set TYPESAFE_API_KEY. TYPESAFE_BASE_URL overrides the host for gateways that proxy the native shape.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ..spec import Question
from . import Provider, ProviderError

DEFAULT_BASE_URL = "https://api.typesafe.ai"
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class TypeSafeProvider(Provider):
    name = "typesafe"

    def __init__(self, model: str = "jev-latest", api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 5, timeout: float = 60.0) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError("TYPESAFE_API_KEY is not set. Use --provider sim to try jevcal without a key.")

    def cache_key(self) -> dict:
        return {"provider": self.name, "model": self.model, "base_url": self.base_url}

    def ask(self, state: Any, questions: dict[str, Question], row: dict | None = None) -> dict:
        body = {
            "state": state,
            "model": self.model,
            "questions": {qid: q.to_api() for qid, q in questions.items()},
        }
        started = time.perf_counter()
        raw = self._post("/v1/systemone", body)
        latency_ms = (time.perf_counter() - started) * 1000
        return normalize_response(raw, questions, latency_ms)

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.base_url + path,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "jevcal",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                if exc.code in RETRY_STATUSES and attempt < self.max_retries:
                    retry_after = exc.headers.get("retry-after")
                    delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2**attempt
                    time.sleep(min(delay, 30))
                    continue
                raise ProviderError(f"TypeSafe API returned {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise ProviderError(f"TypeSafe API unreachable: {exc}") from exc
        raise ProviderError("TypeSafe API: retries exhausted")


def normalize_response(raw: dict, questions: dict[str, Question], latency_ms: float) -> dict:
    answers = {}
    for qid, question in questions.items():
        answer = (raw.get("answers") or {}).get(qid)
        if answer is None:
            raise ProviderError(f"response has no answer for question {qid!r}")
        answers[qid] = normalize_answer(question, answer)
    return {
        "model": raw.get("model"),
        "answers": answers,
        "usage": raw.get("usage") or {},
        "latency_ms": latency_ms,
    }


def normalize_answer(question: Question, answer: dict) -> dict:
    if question.type == "noul":
        p = float(answer["noul"])
        return {
            "type": "noul",
            "answer": p >= 0.5,
            "probabilities": {"true": p, "false": 1.0 - p},
            "confidence": None,  # the API reports no confidence for nouls
        }
    probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
    if question.type == "choice":
        chosen = answer.get("choice") or max(probabilities, key=probabilities.get)
        return {
            "type": "choice",
            "answer": str(chosen),
            "probabilities": probabilities,
            "confidence": answer.get("confidence"),
        }
    top_level = max(probabilities, key=probabilities.get)
    return {
        "type": "score",
        "answer": int(top_level),
        "score": answer.get("score"),
        "probabilities": probabilities,
        "confidence": answer.get("confidence"),
    }
