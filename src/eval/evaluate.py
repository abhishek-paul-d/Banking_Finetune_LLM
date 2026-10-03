"""Evaluate a model served over an OpenAI-compatible HTTP API (llama.cpp server, vLLM) on the test set.

Raw-prompt mode (--tokenizer): sends the EXACT prompt text used in training/eval (thinking off, same
template) to /v1/completions. This avoids any difference in how each server applies the chat template,
so scores are comparable with the HF-based evaluation. Without --tokenizer it uses /v1/chat/completions.

Writes a metrics JSON and a per-example predictions file (for paired comparisons).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from src.eval.metrics import score_rows


def query_chat(base_url, model, msgs, max_tokens=64):
    r = requests.post(f"{base_url}/v1/chat/completions", timeout=300,
                      json={"model": model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def query_raw(base_url, model, prompt, max_tokens=64):
    r = requests.post(f"{base_url}/v1/completions", timeout=300,
                      json={"model": model, "prompt": prompt, "temperature": 0, "max_tokens": max_tokens})
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def evaluate(base_url, model, rows, concurrency, tokenizer_dir=None):
    if tokenizer_dir:
        from transformers import AutoTokenizer

        from src.formatting import build_renderer

        rend = build_renderer(AutoTokenizer.from_pretrained(tokenizer_dir))
        fn = lambda r: query_raw(base_url, model, rend.prompt(r["prompt"]))
    else:
        fn = lambda r: query_chat(base_url, model, r["prompt"])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        texts = [t.strip() for t in ex.map(fn, rows)]
    wall = time.perf_counter() - t0

    metrics, recs = score_rows([r["label"] for r in rows], texts)
    for r, rec in zip(rows, recs):
        rec["text"] = r["prompt"][1]["content"]
    metrics.update(wall_seconds=round(wall, 1), examples_per_s=round(len(rows) / wall, 2),
                   concurrency=concurrency)
    return metrics, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tokenizer", default="", help="dir of the HF tokenizer -> raw-prompt mode (recommended)")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--preds-out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    metrics, recs = evaluate(args.base_url, args.model, rows, args.concurrency, args.tokenizer or None)
    metrics.update(tag=args.tag, name=args.model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    if args.preds_out:
        with open(args.preds_out, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
    print(json.dumps(metrics, indent=2))
    for rec in recs[:3]:
        print("EXAMPLE:", rec["gold"], "|", rec["raw"][:120])


if __name__ == "__main__":
    main()
