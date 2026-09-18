"""Regenerate the bundled example dataset. Synthetic, labeled by construction, deterministic.

    python scripts/make_example.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src/jevcal/examples/support/tickets.jsonl"

ISSUES = {
    "billing": [
        "I was charged twice for my subscription this month",
        "my invoice shows the wrong company address",
        "the refund you promised has still not reached my card",
        "I need a copy of last quarter's receipts for our accountant",
        "my payouts have been failing since the weekend",
        "you billed me for seats we removed weeks ago",
    ],
    "technical": [
        "the dashboard throws a 500 error whenever I open reports",
        "our webhook deliveries stopped arriving",
        "the mobile app logs me out every few minutes",
        "CSV export produces an empty file",
        "the API returns stale data after an update",
        "single sign-on redirects in a loop",
    ],
    "sales": [
        "we want to move from the starter plan to enterprise",
        "can you send pricing for 200 additional seats",
        "is there a discount for annual billing for nonprofits",
        "we would like a demo of the analytics add-on",
        "what is included in the premium support tier",
        "we are comparing vendors and need a security questionnaire filled in",
    ],
}
URGENT = [
    "This is blocking our whole team right now.",
    "We go live tomorrow morning and need this fixed today.",
    "Customers are affected as we speak, please respond immediately.",
    "We are losing money every hour this continues.",
]
CALM_TIMING = [
    "No rush, whenever someone has a moment.",
    "This can wait until next week.",
    "Just flagging it for when you get a chance.",
    "",
]
TONE = {
    0: ["Thanks for your help!", "Appreciate the great product.", "Hope you are having a good week.", ""],
    1: ["This is getting frustrating.", "I have already written about this once.", "I expected this to work by now.",
        "Honestly this is annoying."],
    2: ["This is completely unacceptable.", "I am furious, this is the third time.", "Fix this or we are cancelling our contract.",
        "Worst support experience I have ever had."],
}
OPENERS = ["Hi team,", "Hello,", "Hey,", "To whom it may concern,", "Support,", ""]


def main() -> None:
    rng = random.Random(20260918)
    rows = []
    for index in range(1, 401):
        department = rng.choice(list(ISSUES))
        urgent = rng.random() < 0.4
        frustration = rng.choices([0, 1, 2], weights=[5, 3, 2])[0]
        if urgent:
            frustration = max(frustration, 1)  # "respond immediately" never reads as calm, so don't label it calm
        issue = rng.choice(ISSUES[department])
        body = [issue[0].upper() + issue[1:] + ".",
                rng.choice(URGENT if urgent else CALM_TIMING), rng.choice(TONE[frustration])]
        rng.shuffle(body)
        rows.append({
            "id": f"t{index:03d}",
            "state": " ".join(p for p in [rng.choice(OPENERS), *body] if p).strip(),
            "labels": {"is_urgent": urgent, "department": department, "frustration": frustration},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
