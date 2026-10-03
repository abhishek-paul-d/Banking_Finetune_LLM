"""CI eval gate: exit non-zero if any metric is below its floor (or above its ceiling).

Prints a table, and (when running in GitHub Actions) appends it to the job summary.
"""
import argparse
import json
import os
import sys

import yaml


def check(metrics: dict, thresholds: dict):
    """Return (rows, passed). A missing metric counts as a failure."""
    rows, passed = [], True
    for kind, op in (("min", lambda v, t: v >= t), ("max", lambda v, t: v <= t)):
        for name, limit in (thresholds.get(kind) or {}).items():
            value = metrics.get(name)
            ok = value is not None and op(value, limit)
            passed &= ok
            rows.append((name, value, ">=" if kind == "min" else "<=", limit, ok))
    return rows, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/eval.json")
    ap.add_argument("--thresholds", default="configs/gate.yaml")
    args = ap.parse_args()

    metrics = json.load(open(args.results))
    thresholds = yaml.safe_load(open(args.thresholds))
    rows, passed = check(metrics, thresholds)

    lines = ["| metric | value | required | result |", "|---|---|---|---|"]
    for name, value, sym, limit, ok in rows:
        v = "missing" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {v} | {sym} {limit} | {'PASS' if ok else '**FAIL**'} |")
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {v} (required {sym} {limit})")
    head = f"### Eval gate: {'PASSED' if passed else 'FAILED'} (n={metrics.get('n', '?')}, model={metrics.get('tag') or metrics.get('name', '?')})"
    print(head)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(head + "\n\n" + "\n".join(lines) + "\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
