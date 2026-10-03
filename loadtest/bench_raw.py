"""Concurrency sweep against an OpenAI-compatible /v1/completions server (vLLM, llama.cpp server).

Sends the EXACT prompt text used in training and evaluation (rendered once with thinking off), so every
server sees identical input regardless of how it applies chat templates. For each concurrency level it
records latency percentiles, requests/s and generated tokens/s, then appends rows to a CSV.
Re-running a label replaces that label's old rows.
"""
import argparse
import asyncio
import csv
import json
import math
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `src.*` imports work when run as a script


async def one(client, url, model, prompt, sem, max_tokens):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/v1/completions", json={
                "model": model, "prompt": prompt, "temperature": 0, "max_tokens": max_tokens})
            r.raise_for_status()
            u = r.json().get("usage", {})
            return time.perf_counter() - t0, u.get("completion_tokens", 0), u.get("prompt_tokens", 0), None
        except Exception as e:  # noqa: BLE001
            return time.perf_counter() - t0, 0, 0, repr(e)[:120]


async def run_level(url, model, prompts, conc, n, max_tokens):
    sem = asyncio.Semaphore(conc)
    reqs = [prompts[i % len(prompts)] for i in range(n)]
    limits = httpx.Limits(max_connections=conc + 8, max_keepalive_connections=conc + 8)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        t0 = time.perf_counter()
        out = await asyncio.gather(*(one(client, url, model, p, sem, max_tokens) for p in reqs))
        wall = time.perf_counter() - t0
    ok = [o for o in out if o[3] is None]
    errors = len(out) - len(ok)
    if not ok:
        return {"concurrency": conc, "requests": n, "errors": errors}
    lat = sorted(o[0] for o in ok)
    pct = lambda q: lat[max(0, math.ceil(q * len(lat)) - 1)]          # nearest-rank percentile
    toks = sum(o[1] for o in ok)
    return {
        "concurrency": conc, "requests": n, "errors": errors, "wall_s": round(wall, 2),
        "req_per_s": round(len(ok) / wall, 2), "out_tok_per_s": round(toks / wall, 1),
        "p50_s": round(pct(0.50), 4), "p95_s": round(pct(0.95), 4), "p99_s": round(pct(0.99), 4),
        "mean_out_tokens": round(toks / len(ok), 1), "mean_prompt_tokens": round(sum(o[2] for o in ok) / len(ok), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="banking", help="served model name")
    ap.add_argument("--label", required=True, help="variant name for the CSV, e.g. vllm-awq-default")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir or id used to render the prompts")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--prompts", type=int, default=500, help="number of distinct test prompts to cycle through")
    ap.add_argument("--levels", default="1,4,8,16,32,64,128")
    ap.add_argument("--min-requests", type=int, default=300)
    ap.add_argument("--requests-per-conc", type=int, default=6, help="requests per level = max(min, this x concurrency)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out-csv", default="results/phase4/sweep.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from src.formatting import build_renderer

    rend = build_renderer(AutoTokenizer.from_pretrained(args.tokenizer))
    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")][: args.prompts]
    prompts = [rend.prompt(r["prompt"]) for r in rows]

    asyncio.run(run_level(args.base_url, args.model, prompts, 8, 48, args.max_tokens))   # warm-up, discarded
    results = []
    for conc in map(int, args.levels.split(",")):
        n = max(args.min_requests, args.requests_per_conc * conc)
        res = asyncio.run(run_level(args.base_url, args.model, prompts, conc, n, args.max_tokens))
        res["variant"] = args.label
        results.append(res)
        print(res, flush=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if out.exists():
        old = [r for r in csv.DictReader(open(out, newline="", encoding="utf-8")) if r["variant"] != args.label]
    fields = ["variant", "concurrency", "requests", "errors", "wall_s", "req_per_s", "out_tok_per_s",
              "p50_s", "p95_s", "p99_s", "mean_out_tokens", "mean_prompt_tokens"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(old)
        w.writerows(results)


if __name__ == "__main__":
    main()
