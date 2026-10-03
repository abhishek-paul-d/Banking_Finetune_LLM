"""Build chat-format JSONL splits from Banking77.

Full splits:   train / val / test
Pilot splits:  pilot_train / pilot_val (stratified, disjoint) and test_small
               (stratified slice of test) for the Phase 1 model comparison.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.schema import SYSTEM_PROMPT, make_target


def to_record(text: str, intent: str) -> dict:
    target = make_target(intent)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "label": target,
    }


GH = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


def load_banking77():
    """Return (train, test) as lists of (text, intent_name).

    Tries the HF repo PolyAI/banking77 first. That repo is a loading script, which
    `datasets` >= 4 refuses to run, so we fall back to the exact CSVs the script
    downloads (same data, same labels).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
        names = ds["train"].features["label"].names
        return (
            [(r["text"], names[r["label"]]) for r in ds["train"]],
            [(r["text"], names[r["label"]]) for r in ds["test"]],
        )
    except Exception as e:
        print(f"HF load failed ({type(e).__name__}); using the CSVs behind PolyAI/banking77")
        import pandas as pd

        out = []
        for split in ("train", "test"):
            df = pd.read_csv(f"{GH}/{split}.csv")
            out.append(list(zip(df["text"], df["category"])))
        return tuple(out)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def take_per_class(rows, n: int):
    """Split `rows` into (first n per intent, the rest). `rows` must be pre-shuffled."""
    seen = defaultdict(int)
    picked, rest = [], []
    for r in rows:
        k = r["label"]["intent"]
        if seen[k] < n:
            seen[k] += 1
            picked.append(r)
        else:
            rest.append(r)
    return picked, rest


def summarize(name: str, rows) -> dict:
    intents = Counter(r["label"]["intent"] for r in rows)
    urgency = Counter(r["label"]["urgency"] for r in rows)
    human = Counter(r["label"]["needs_human"] for r in rows)
    return {
        "split": name,
        "n": len(rows),
        "n_intents": len(intents),
        "min_per_intent": min(intents.values()),
        "max_per_intent": max(intents.values()),
        "urgency": dict(urgency),
        "needs_human": {str(k): v for k, v in human.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--pilot-per-class", type=int, default=26)
    ap.add_argument("--pilot-val-per-class", type=int, default=5)
    ap.add_argument("--test-small-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    raw_train, raw_test = load_banking77()
    train_all = [to_record(t, i) for t, i in raw_train]
    test = [to_record(t, i) for t, i in raw_test]
    rng.shuffle(train_all)
    rng.shuffle(test)

    # Full splits
    n_val = int(len(train_all) * args.val_frac)
    val, train = train_all[:n_val], train_all[n_val:]

    # Pilot splits: drawn from the full train split, disjoint from each other.
    # Pilot val is also kept out of the full `test`, which is never touched.
    pilot_train, rest = take_per_class(train_all, args.pilot_per_class)
    pilot_val, _ = take_per_class(rest, args.pilot_val_per_class)

    # Stratified slice of test for fast eval
    test_small, _ = take_per_class(test, args.test_small_per_class)

    splits = {
        "train": train, "val": val, "test": test,
        "pilot_train": pilot_train, "pilot_val": pilot_val, "test_small": test_small,
    }
    stats = []
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        stats.append(summarize(name, rows))

    # Leakage check: no pilot/test text overlap
    texts = lambda rows: {r["prompt"][1]["content"] for r in rows}
    assert not texts(pilot_train) & texts(test), "pilot_train overlaps test"
    assert not texts(pilot_train) & texts(pilot_val), "pilot_train overlaps pilot_val"

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    for s in stats:
        print(f"{s['split']:<12} n={s['n']:<6} intents={s['n_intents']} "
              f"per-intent={s['min_per_intent']}-{s['max_per_intent']} urgency={s['urgency']}")


if __name__ == "__main__":
    main()
