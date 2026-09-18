# jevcal

**Stop guessing confidence thresholds.** jevcal measures a typed decision model on *your* data, picks the
threshold that meets *your* accuracy target, tells you how much traffic still needs an LLM, and fails CI
when a model update quietly breaks it.

Built for [Jev](https://typesafe.ai) (TypeSafe's System One model), and for anything else that returns
answers with probabilities.

```
$ jevcal demo        # bundled example, built-in simulator, no API key

question               threshold  handled  accepted acc  all acc     ECE  status
is_urgent                  0.994    21.2%        100.0%    94.5%    3.5%  ok
department                 0.954    81.3%         96.3%    93.8%    3.5%  ok
frustration                0.875    98.0%         95.4%    95.5%    1.6%  ok
(handled / accepted accuracy are measured on the held-out split)

rows that escalate: 85.4%   cascade $1.927 per 1k rows vs LLM-only $2.25 (14.4% saved)
a row escalates when any question is unsure; the bottleneck is is_urgent (handles 21.2% at a 97% target)
```

That is simulator output, not a Jev benchmark. Read it as the kind of answer you get: `department` can run
81% on the fast model at 96% accuracy, while the 97% target on `is_urgent` is so strict that it alone sends
most rows to the LLM. Loosen that one target, or reword that one question, and the bill moves.

## Why

Jev answers "is this fraud?" with a probability, not a paragraph. That is the whole appeal: you can act on
the confident answers and send the rest to a slower, smarter model. But *where is the line?*

- Pick it too high and most traffic escalates, so the savings disappear.
- Pick it too low and you act on wrong answers.
- The vendor docs say the right value "depends on your domain" and to "test with your own data".
- `jev-latest` is an alias that moves on every release, so a threshold tuned today can drift tomorrow.

jevcal is the "test with your own data" part, as one command.

## Install

```bash
pip install "git+https://github.com/abhixhek/jevcal"                        # core
pip install "jevcal[anthropic] @ git+https://github.com/abhixhek/jevcal"    # + Claude as teacher / fallback
```

Not on PyPI yet.

Python 3.10+.

## Try it in 10 seconds (no API key)

```bash
jevcal demo
open jevcal-demo/report.html
```

The demo runs against a built-in **simulator**, not Jev. It exists so you can see the workflow and the
report before you have a key. The simulator is deliberately overconfident so there is something to find.

## Real workflow

**1. Describe your decisions** in `questions.yaml` (same shape as the TypeSafe API):

```yaml
model: jev-1.13.0
questions:
  is_fraud:
    type: noul
    instructions: Is this email a fraud attempt?
    criteria:
      "true": Asks for credentials, payment, or urgent action under false pretenses.
      "false": Ordinary correspondence.
    target: 0.99
  queue:
    type: choice
    instructions: Which team should handle this ticket?
    criteria: { billing: Charges and refunds, technical: Bugs and outages, sales: Pricing and plans }
```

**2. Bring data** as JSONL, a few hundred real examples: `{"id": "1", "state": "...", "labels": {"is_fraud": true}}`.
No labels yet? Let an LLM teacher fill them in (it never overwrites labels you already have):

```bash
export ANTHROPIC_API_KEY=...
jevcal label --questions questions.yaml --data raw.jsonl --out data.jsonl --llm anthropic:claude-opus-5
```

LLM labels are a teacher signal, not ground truth. Spot-check a sample.

**3. Lint the wording** against the model's documented weak spots (negations, counting, dates, compound
questions, overlapping options). No API key needed:

```bash
jevcal lint --questions questions.yaml --data data.jsonl
```

**4. Measure and compile:**

```bash
export TYPESAFE_API_KEY=...
jevcal run --questions questions.yaml --data data.jsonl --target 0.98
```

You get `decisions.lock.json` (thresholds + the evidence behind them) and `jevcal-report.html`
(reliability diagram, accuracy-vs-coverage curve, per-answer accuracy, cost split).

Thresholds are **picked on one half of your data and verified on the other half**. If a threshold does
not hold on the held-out half, jevcal says so instead of reporting the flattering number. Add
`--conservative` to require the 95% lower confidence bound, not the point estimate, to clear the target.

**5. Let the LLM fix badly worded questions** (optional). All rewrites are evaluated in a single request
per row, because extra questions are nearly free on Jev. A rewrite is kept only if the held-out split
agrees it handles more traffic at your target:

```bash
jevcal optimize --questions questions.yaml --data data.jsonl --llm anthropic:claude-opus-5
```

**6. Ship the cascade:**

```python
from jevcal.runtime import Cascade, llm_fallback

cascade = Cascade.from_lock(
    "decisions.lock.json",
    fallback=llm_fallback("anthropic:claude-opus-5"),
    log_path="decisions.jsonl",   # escalations where the LLM disagrees are your next eval rows
)
decisions = cascade.decide(ticket_text)
decisions["queue"].answer    # "billing"
decisions["queue"].source    # "jev" | "fallback" | "unresolved"
```

With no fallback configured it fails closed: low-confidence answers come back as `None`, never as a guess.

**7. Guard it in CI:**

```bash
jevcal check --lock decisions.lock.json --data data.jsonl   # exit 1 on drift
```

Fails when accepted accuracy drops below target, when coverage falls (more traffic escalating), or, with
`--strict`, when the model version that answered is not the one the thresholds were tuned on.

## Commands

| Command | What it does | Needs a key |
|---|---|---|
| `lint` | Flags wording the model is known to handle badly | no |
| `label` | LLM teacher fills missing gold labels | LLM |
| `measure` | Runs the model over the dataset, saves predictions | Jev |
| `compile` | Picks thresholds from saved predictions, writes lock + report | no |
| `run` | `measure` + `compile` | Jev |
| `optimize` | LLM rewrites questions, held-out data decides | both |
| `check` | Re-measures against the lock, exits 1 on drift | Jev |
| `demo` | Whole flow against the simulator | no |

`--provider llm --llm openai:gpt-...` runs the same questions through a text LLM, so you can compare an
all-LLM baseline against your labels. LLM specs: `anthropic:<model>`, `openai:<model>` (honours
`OPENAI_BASE_URL`, so any compatible endpoint works), `openrouter:<model>`.

## What the numbers mean

- **Accepted accuracy**: accuracy among decisions the fast model keeps (confidence at or above threshold).
- **Handled**: share of decisions it keeps. The rest escalate.
- **ECE**: average gap between stated confidence and observed accuracy. 0 means "90% sure" is right 90% of the time.
- **Confidence measure**: `top_prob` by default. `margin`, `entropy`, and the API's own `confidence` are
  also scored; `--measure auto` switches only when another one separates right from wrong clearly better.

## Honest limits

- A threshold is only as good as your sample. Under ~100 labeled rows per question, expect it not to hold.
- One threshold per question. Per-answer accuracy is reported so you can see when one class is weak.
- Responses are cached in `.jevcal/cache` so reruns are free; `check` always bypasses the cache.
- Tested end to end against the simulator and against the response shapes in TypeSafe's public API reference.
  Not affiliated with TypeSafe AI.
- Measure your own data, privately. jevcal ships no leaderboard and publishes nothing.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest
python scripts/make_example.py   # regenerate the bundled example dataset
```

MIT licensed.
