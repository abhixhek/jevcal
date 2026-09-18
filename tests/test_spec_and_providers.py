import json

import pytest

from jevcal.providers.llm import CallableLLM, LLMProvider, extract_json
from jevcal.providers.typesafe import normalize_response
from jevcal.spec import Question, SpecError, load_questions

NOUL = Question("is_urgent", "noul", "Does this convey urgency?")
CHOICE = Question("department", "choice", "Which team?", {"billing": "x", "technical": "y", "sales": None})
SCORE = Question("frustration", "score", "How frustrated?", ["Calm", "Frustrated", "Very angry"])


def test_normalize_answers():
    assert NOUL.normalize("Yes") is True and NOUL.normalize(False) is False
    assert CHOICE.normalize("Billing") == "billing"
    assert SCORE.normalize(2) == 2 and SCORE.normalize("1") == 1 and SCORE.normalize("very angry") == 2
    for question, bad in ((NOUL, "maybe"), (CHOICE, "legal"), (SCORE, 7), (SCORE, True)):
        with pytest.raises(SpecError):
            question.normalize(bad)


def test_yaml_boolean_criteria_keys_become_strings(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("questions:\n  u:\n    type: noul\n    instructions: Is it urgent today?\n    criteria:\n      true: yes it is\n      false: no it is not\n")
    question = load_questions(path).questions["u"]
    assert set(question.criteria) == {"true", "false"}
    assert question.to_api()["criteria"]["true"] == "yes it is"


def test_invalid_questions_rejected():
    with pytest.raises(SpecError):
        Question("c", "choice", "pick", {"only": "one"})
    with pytest.raises(SpecError):
        Question("s", "score", "rate", ["single"])
    with pytest.raises(SpecError):
        Question("x", "essay", "write")


def test_typesafe_response_normalization_matches_documented_shapes():
    # shapes copied from https://docs.typesafe.ai/api
    raw = {
        "model": "jev-1.13.0",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.92},
            "department": {"type": "choice", "choice": "technical",
                           "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07}, "confidence": 0.82},
            "frustration": {"type": "score", "score": 1.6, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                            "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65}, "confidence": 0.78},
        },
        "usage": {"input_tokens": 312, "output_tokens": 48},
    }
    out = normalize_response(raw, {q.id: q for q in (NOUL, CHOICE, SCORE)}, latency_ms=100)
    assert out["model"] == "jev-1.13.0"
    assert out["answers"]["is_urgent"]["answer"] is True
    assert out["answers"]["is_urgent"]["probabilities"] == {"true": 0.92, "false": pytest.approx(0.08)}
    assert out["answers"]["department"]["answer"] == "technical" and out["answers"]["department"]["confidence"] == 0.82
    assert out["answers"]["frustration"]["answer"] == 2


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": true}\n```') == {"a": True}
    assert extract_json('Here you go: {"a": "b"} hope it helps') == {"a": "b"}
    assert extract_json("[1, 2]") == [1, 2]


def test_llm_provider_retries_once_on_bad_reply():
    replies = iter(["I think it is urgent.", json.dumps({"is_urgent": "yes", "department": "sales", "frustration": 0})])
    provider = LLMProvider(CallableLLM(lambda system, user: next(replies)))
    out = provider.ask("some ticket", {q.id: q for q in (NOUL, CHOICE, SCORE)})
    assert out["answers"]["is_urgent"]["answer"] is True
    assert out["answers"]["department"]["probabilities"] == {"billing": 0.0, "technical": 0.0, "sales": 1.0}
    assert out["answers"]["frustration"]["answer"] == 0
