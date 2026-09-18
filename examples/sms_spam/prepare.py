"""Convert the UCI SMS Spam Collection into jevcal's JSONL format.

Get the data first (CC BY 4.0, ~200 KB):
    https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Unzip it, then:
    python examples/sms_spam/prepare.py path/to/SMSSpamCollection --sample 1000

The file is tab-separated: <ham|spam>\t<message>. The sample keeps the original class balance.
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to the SMSSpamCollection file")
    parser.add_argument("--out", default="examples/sms_spam/data.jsonl")
    parser.add_argument("--sample", type=int, default=0, help="rows to keep (0 = all)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows, seen = [], set()
    for line in Path(args.source).read_text(encoding="utf-8", errors="replace").splitlines():
        label, _, text = line.partition("\t")
        text = text.strip()
        if label not in {"ham", "spam"} or not text or text in seen:
            continue  # the collection contains exact duplicates; they would leak across the fit/held-out split
        seen.add(text)
        rows.append({"state": text, "labels": {"is_spam": label == "spam"}})

    if args.sample and args.sample < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    for index, row in enumerate(rows, 1):
        row["id"] = f"sms{index:05d}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"id": r["id"], "state": r["state"], "labels": r["labels"]}, ensure_ascii=False) + "\n" for r in rows))
    spam = sum(r["labels"]["is_spam"] for r in rows)
    print(f"wrote {len(rows)} rows ({spam} spam, {len(rows) - spam} ham) to {out}")


if __name__ == "__main__":
    main()
