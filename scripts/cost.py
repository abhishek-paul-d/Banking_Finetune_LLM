"""Cost per 1M output tokens from measured throughput and instance price.

cost_per_1M = hourly_price / (tokens_per_s * 3600) * 1e6

Usage: python scripts/cost.py --hourly 0.526 --csv results/bench.csv
(0.526 is g4dn.xlarge on-demand us-east-1; check current pricing.)
Uses the highest tokens/s level per variant, i.e. best-case utilization.
Compare against your hosted API's output price per 1M tokens.
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", type=float, required=True, help="instance $/hour")
    ap.add_argument("--csv", default="results/bench.csv")
    args = ap.parse_args()

    best = defaultdict(float)
    for r in csv.DictReader(open(args.csv)):
        best[r["variant"]] = max(best[r["variant"]], float(r["tokens_per_s"]))

    print(f"{'variant':<16}{'peak tok/s':>12}{'$/1M tokens':>14}")
    for v, tps in sorted(best.items()):
        print(f"{v:<16}{tps:>12.1f}{args.hourly / (tps * 3600) * 1e6:>14.3f}")


if __name__ == "__main__":
    main()
