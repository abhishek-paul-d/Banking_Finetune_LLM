"""Evaluate a base model (zero-shot) or base + LoRA adapter on a split, locally with HF generate.

Uses the exact prompt rendering from training (src.formatting), greedy decoding,
and records per-example predictions so models can be compared with a paired test.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.config import load_cfg
from src.eval.metrics import score_rows
from src.formatting import build_renderer
from src.modeling import extra_input_keys, load_model
from src.schema import SYSTEM_PROMPT, zero_shot_system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="data/test.jsonl")
    ap.add_argument("--adapter", default="", help="adapter dir; empty = base model only")
    ap.add_argument("--zero-shot", action="store_true", help="list the 77 labels in the system prompt")
    ap.add_argument("--tag", required=True, help="e.g. finetuned_test or zeroshot_small")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels-from", default="data/train.jsonl")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.split, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.padding_side = "left"
    rend = build_renderer(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    sys_override = None
    if args.zero_shot:
        intents = {json.loads(l)["label"]["intent"] for l in open(args.labels_from, encoding="utf-8")}
        sys_override = zero_shot_system(intents)

    def prompt_for(row):
        msgs = [dict(m) for m in row["prompt"]]
        if sys_override:
            assert msgs[0]["content"] == SYSTEM_PROMPT
            msgs[0]["content"] = sys_override
        return rend.prompt(msgs)

    prompts = [prompt_for(r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # similar lengths per batch

    model = load_model(cfg["base_model"], cfg["load_in_4bit"], cfg.get("loaders"), args.adapter or None,
                       decompress=cfg.get("decompress_compressed_tensors", False))
    model.eval()
    extra = extra_input_keys(model)
    device = next(model.parameters()).device

    texts = [None] * len(rows)
    new_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for s in range(0, len(order), args.batch_size):
        idx = order[s : s + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        inputs = dict(enc)
        for k in extra:
            inputs[k] = torch.zeros_like(enc["input_ids"])
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=rend.end_ids, pad_token_id=pad_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        for i, g in zip(idx, gen):
            texts[i] = tok.decode(g, skip_special_tokens=True).strip()
            new_tokens += int((g != pad_id).sum())
        if (s // args.batch_size) % 10 == 0:
            print(f"{s + len(idx)}/{len(rows)}", flush=True)
    wall = time.time() - t0

    metrics, recs = score_rows([r["label"] for r in rows], texts)
    for r, rec in zip(rows, recs):
        rec["text"] = r["prompt"][1]["content"]
    metrics.update(
        tag=args.tag, name=cfg["name"], wall_seconds=round(wall, 1),
        gen_tokens_per_s=round(new_tokens / wall, 1),
        examples_per_s=round(len(rows) / wall, 2),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
    )
    (out_dir / f"{args.tag}.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / f"{args.tag}_preds.jsonl").open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(metrics, indent=2))
    for rec in recs[:3]:
        print("EXAMPLE:", rec["gold"], "|", rec["raw"][:120])


if __name__ == "__main__":
    main()
