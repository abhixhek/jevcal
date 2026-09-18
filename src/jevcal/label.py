"""Fill in gold labels with an LLM teacher."""

from __future__ import annotations

from .measure import run
from .providers.llm import LLM, LLMProvider
from .spec import QuestionSet


def label_rows(questions: QuestionSet, rows: list[dict], llm: LLM, *, overwrite: bool = False,
               concurrency: int = 4, quiet: bool = False) -> dict:
    """Mutates `rows` in place. Returns counts. Human labels are never overwritten unless asked."""
    wanted = set(questions.questions)
    todo = [r for r in rows if overwrite or not wanted <= set(r["labels"])]
    predictions = run(LLMProvider(llm), questions.questions, todo, concurrency=concurrency, quiet=quiet)
    by_id = {p["id"]: p for p in predictions}
    labeled, failed = 0, []
    for row in todo:
        prediction = by_id[row["id"]]
        if "error" in prediction:
            failed.append({"id": row["id"], "error": prediction["error"]})
            continue
        for qid, answer in prediction["answers"].items():
            if overwrite or qid not in row["labels"]:
                row["labels"][qid] = answer["answer"]
        row["labeled_by"] = llm.spec
        labeled += 1
    return {"labeled": labeled, "skipped": len(rows) - len(todo), "failed": failed}
