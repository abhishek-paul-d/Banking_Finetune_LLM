"""Cost of serving, from measured peak throughput and a GPU price.

cost per 1M requests      = hourly / (req_per_s * 3600) * 1e6
cost per 1M output tokens = hourly / (out_tok_per_s * 3600) * 1e6

Uses each variant's best level in the concurrency sweep, i.e. the GPU is assumed 100% busy. Real traffic is burstier,
so treat these as the floor of what the hardware costs; divide by your expected utilization for a realistic figure.

Usage: python scripts/cost.py --sweep results/phase4/sweep.csv --hourly 0.156 --hourly 1.00 --out results/phase4/cost_table.md
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/phase4/sweep.csv")
    ap.add_argument("--hourly", type=float, action="append", required=True, help="GPU $/hour (repeat for several prices)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    best = defaultdict(lambda: {"req": 0.0, "tok": 0.0, "conc": 0})
    for r in csv.DictReader(open(args.sweep, encoding="utf-8")):
        b = best[r["variant"]]
        if float(r["req_per_s"]) > b["req"]:
            b.update(req=float(r["req_per_s"]), tok=float(r["out_tok_per_s"]), conc=int(r["concurrency"]))

    price_cols = " | ".join(f"$/1M req @ ${h:g}/h" for h in args.hourly)
    lines = [f"| variant | peak req/s | at concurrency | peak out tok/s | {price_cols} | $/1M out tokens @ ${args.hourly[0]:g}/h |",
             "|" + "---|" * (4 + len(args.hourly) + 1)]
    for v, b in best.items():
        per_req = " | ".join(f"{h / (b['req'] * 3600) * 1e6:.2f}" for h in args.hourly)
        per_tok = args.hourly[0] / (b["tok"] * 3600) * 1e6
        lines.append(f"| {v} | {b['req']:.1f} | {b['conc']} | {b['tok']:.0f} | {per_req} | {per_tok:.2f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
