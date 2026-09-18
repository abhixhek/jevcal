"""LLM backends: the teacher that labels data, rewrites questions, and handles escalations.

Spec strings:
    anthropic:claude-opus-5          official `anthropic` SDK (pip install 'jevcal[anthropic]')
    openai:<model>                   any OpenAI-compatible endpoint (OPENAI_API_KEY, OPENAI_BASE_URL)
    openrouter:<model>               OpenRouter (OPENROUTER_API_KEY)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from ..spec import Question, SpecError, state_text
from . import Provider, ProviderError

DEFAULT_LLM = "anthropic:claude-opus-5"

LABEL_SYSTEM = (
    "You label data for a decision-model evaluation. Read the state and answer every question "
    "exactly as its instructions and criteria define it. Reply with one JSON object mapping each "
    "question id to its answer and nothing else. Answer formats: noul -> true or false; "
    "choice -> one option key, verbatim; score -> the integer index of the best level (0-based)."
)


class LLMRefusal(ProviderError):
    pass


class LLM:
    spec = "llm"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("the anthropic SDK is not installed: pip install 'jevcal[anthropic]'") from exc
        self._anthropic = anthropic
        self.model = model
        self.spec = f"anthropic:{model}"
        self.client = anthropic.Anthropic()  # credentials resolve from the environment

    def complete(self, system: str, user: str) -> str:
        anthropic = self._anthropic
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as exc:
            raise ProviderError(f"Anthropic rate limit hit after SDK retries: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API returned {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Anthropic API unreachable: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMRefusal("the model declined this request")
        # Thinking blocks may precede the answer; only text blocks carry it.
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAICompatLLM(LLM):
    def __init__(self, model: str, base_url: str, api_key: str | None, label: str) -> None:
        if not api_key:
            raise ProviderError(f"no API key found for {label}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.spec = f"{label}:{model}"

    def complete(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
        ).encode()
        for attempt in range(5):
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read())
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise ProviderError(f"{self.spec} returned {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise ProviderError(f"{self.spec} unreachable: {exc}") from exc
        raise ProviderError(f"{self.spec}: retries exhausted")


class CallableLLM(LLM):
    """Wrap any function (system, user) -> str. Used in tests and for custom backends."""

    def __init__(self, fn: Callable[[str, str], str], spec: str = "callable") -> None:
        self.fn = fn
        self.spec = spec

    def complete(self, system: str, user: str) -> str:
        return self.fn(system, user)


def build_llm(spec: str | None) -> LLM:
    spec = spec or DEFAULT_LLM
    kind, _, model = spec.partition(":")
    if not model:
        raise ProviderError(f"LLM spec {spec!r} should look like provider:model, e.g. {DEFAULT_LLM}")
    if kind == "anthropic":
        return AnthropicLLM(model)
    if kind == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAICompatLLM(model, base, os.environ.get("OPENAI_API_KEY"), "openai")
    if kind == "openrouter":
        return OpenAICompatLLM(model, "https://openrouter.ai/api/v1", os.environ.get("OPENROUTER_API_KEY"), "openrouter")
    raise ProviderError(f"unknown LLM provider {kind!r} (expected anthropic, openai, or openrouter)")


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model reply, tolerating code fences and prose."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"[\[{]", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
                return value
            except json.JSONDecodeError:
                continue
    raise ProviderError(f"no JSON found in model reply: {text[:200]!r}")


def questions_block(questions: dict[str, Question]) -> str:
    return json.dumps({qid: q.to_api() for qid, q in questions.items()}, indent=2, ensure_ascii=False)


class LLMProvider(Provider):
    """Answers typed questions with a text LLM. Answers are discrete, so confidence is always 1.0."""

    name = "llm"

    def __init__(self, llm: LLM, attempts: int = 2) -> None:
        self.llm = llm
        self.attempts = attempts

    def cache_key(self) -> dict:
        return {"provider": self.name, "llm": self.llm.spec}

    def ask(self, state: Any, questions: dict[str, Question], row: dict | None = None) -> dict:
        user = f"<state>\n{state_text(state)}\n</state>\n\n<questions>\n{questions_block(questions)}\n</questions>"
        started = time.perf_counter()
        problem = ""
        for _ in range(self.attempts):
            reply = self.llm.complete(LABEL_SYSTEM, user + problem)
            try:
                parsed = extract_json(reply)
                if not isinstance(parsed, dict):
                    raise SpecError("reply was not a JSON object")
                answers = {qid: q.normalize(parsed[qid]) for qid, q in questions.items()}
                break
            except (ProviderError, SpecError, KeyError) as exc:
                problem = f"\n\nYour previous reply was rejected ({exc}). Reply with the JSON object only."
        else:
            raise ProviderError(f"LLM never produced valid answers: {problem.strip()}")
        return {
            "model": self.llm.spec,
            "answers": {qid: one_hot(questions[qid], answer) for qid, answer in answers.items()},
            "usage": {},
            "latency_ms": (time.perf_counter() - started) * 1000,
        }


def one_hot(question: Question, answer: Any) -> dict:
    chosen = question.key_of(answer)
    return {
        "type": question.type,
        "answer": answer,
        "probabilities": {key: 1.0 if key == chosen else 0.0 for key in question.options()},
        "confidence": 1.0,
    }
