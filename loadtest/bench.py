"""Concurrency sweep against an OpenAI-compatible endpoint.

Records p50/p95 latency and output tokens/sec per concurrency level -> CSV.
Run once per variant (fp16, AWQ, GGUF) with --label, then compare.
"""
import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import httpx


async def one(client, url, model, msgs, sem):
    async with sem:
        t0 = time.perf_counter()
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"model": model, "messages": msgs, "temperature": 0, "max_tokens": 64},
        )
        r.raise_for_status()
        dt = time.perf_counter() - t0
        return dt, r.json()["usage"]["completion_tokens"]


async def run_level(url, model, prompts, concurrency, n_requests):
    sem = asyncio.Semaphore(concurrency)
    reqs = [prompts[i % len(prompts)] for i in range(n_requests)]
    async with httpx.AsyncClient(timeout=300) as client:
        t0 = time.perf_counter()
        out = await asyncio.gather(*(one(client, url, model, m, sem) for m in reqs))
        wall = time.perf_counter() - t0
    lats = sorted(d for d, _ in out)
    toks = sum(t for _, t in out)
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "p50_s": statistics.median(lats),
        "p95_s": lats[int(0.95 * (len(lats) - 1))],
        "tokens_per_s": toks / wall,
        "req_per_s": n_requests / wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="variant name, e.g. awq-int4")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--levels", default="1,4,8,16,32,64")
    ap.add_argument("--requests-per-level", type=int, default=200)
    ap.add_argument("--out", default="results/bench.csv")
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.test, encoding="utf-8")][:500]
    rows = []
    for c in map(int, args.levels.split(",")):
        row = asyncio.run(run_level(args.base_url, args.model, prompts, c, args.requests_per_level))
        row["variant"] = args.label
        rows.append(row)
        print(row)

    out = Path(args.out)
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
