"""Build the fixed CI gate slice: a deterministic, class-balanced subset of the test set.

Every intent appears about equally often, so a model that has lost a whole group of labels cannot hide.
The slice is committed to the repo (data/gate_slice.jsonl) so CI does not need to download Banking77.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/gate_slice.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]["intent"]].append(r)
    rng = random.Random(args.seed)
    for lst in by_label.values():
        rng.shuffle(lst)
    labels = sorted(by_label)
    rng.shuffle(labels)

    picked, i = [], 0
    while len(picked) < args.n:                      # round-robin over intents, shuffled order
        lab = labels[i % len(labels)]
        k = i // len(labels)
        if k < len(by_label[lab]):
            picked.append(by_label[lab][k])
        i += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} rows ({len({r['label']['intent'] for r in picked})} intents) -> {args.out}")


if __name__ == "__main__":
    main()
