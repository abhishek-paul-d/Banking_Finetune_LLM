"""Phase 1 decision: compare the pilot models and apply the pre-registered rule.

Rule: higher macro-F1 wins (needs JSON-valid >= 0.98). If the paired-bootstrap
95% CI of the macro-F1 difference includes 0, or |diff| < 0.01, it is a tie:
the faster model is preferred and a second seed run is recommended.
"""
import argparse
import json
import random
from pathlib import Path

from sklearn.metrics import f1_score

MIN_VALID = 0.98
TIE_MARGIN = 0.01


def load(root: Path, name: str):
    d = root / name
    ft = json.loads((d / "finetuned_test.json").read_text())
    preds = [json.loads(l) for l in open(d / "finetuned_test_preds.jsonl", encoding="utf-8")]
    zs_path = d / "zeroshot_small.json"
    zs = json.loads(zs_path.read_text()) if zs_path.exists() else None
    zs_preds = [json.loads(l) for l in open(d / "zeroshot_small_preds.jsonl", encoding="utf-8")] if zs else None
    info = json.loads((d / "train_info.json").read_text())
    return {"name": name, "ft": ft, "zs": zs, "info": info, "preds": preds,
            "ft_i": intent_stats(preds), "zs_i": intent_stats(zs_preds) if zs_preds else None}


def macro_f1(preds, idx, labels=None):
    """Macro-F1 over the real (gold) labels only; invented labels are not extra classes."""
    labels = labels or sorted({p["gold"] for p in preds})
    return f1_score([preds[i]["gold"] for i in idx], [preds[i]["pred"] for i in idx],
                    labels=labels, average="macro", zero_division=0)


def intent_stats(preds):
    labels = sorted({p["gold"] for p in preds})
    real = set(labels)
    n = len(preds)
    return {
        "acc": sum(p["gold"] == p["pred"] for p in preds) / n,
        "f1": macro_f1(preds, range(n), labels),
        "invented": sum(1 for p in preds if p["pred"] not in real and p["pred"] != "__invalid__") / n,
    }


def paired_bootstrap(pa, pb, n_boot=300, seed=0):
    rng = random.Random(seed)
    n = len(pa)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(macro_f1(pa, idx) - macro_f1(pb, idx))
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, required=True)
    ap.add_argument("--root", default="outputs/pilot")
    ap.add_argument("--out", default="results/pilot/comparison.md")
    args = ap.parse_args()
    root = Path(args.root)
    a, b = (load(root, m) for m in args.models)
    assert [p["text"] for p in a["preds"]] == [p["text"] for p in b["preds"]], "different eval rows"

    lo, hi = paired_bootstrap(a["preds"], b["preds"])
    everyone = range(len(a["preds"]))
    diff = macro_f1(a["preds"], everyone) - macro_f1(b["preds"], everyone)

    nan = float("nan")
    rows = [
        ("zero-shot macro-F1 (770 ex.)", lambda m: m["zs_i"]["f1"] if m["zs_i"] else nan),
        ("zero-shot intent accuracy", lambda m: m["zs_i"]["acc"] if m["zs_i"] else nan),
        ("zero-shot JSON valid", lambda m: m["zs"]["json_valid_rate"] if m["zs"] else nan),
        ("fine-tuned intent accuracy", lambda m: m["ft_i"]["acc"]),
        ("fine-tuned macro-F1 (77 real labels)", lambda m: m["ft_i"]["f1"]),
        ("invented-label rate", lambda m: m["ft_i"]["invented"]),
        ("fine-tuned JSON valid", lambda m: m["ft"]["json_valid_rate"]),
        ("urgency accuracy", lambda m: m["ft"]["urgency_accuracy"]),
        ("needs_human accuracy", lambda m: m["ft"]["needs_human_accuracy"]),
        ("train time (s)", lambda m: m["info"]["train_seconds"]),
        ("train peak VRAM (GB)", lambda m: m["info"]["peak_vram_gb"]),
        ("final val loss", lambda m: m["info"]["final_val_loss"]),
        ("eval gen tokens/s (bs16)", lambda m: m["ft"]["gen_tokens_per_s"]),
        ("eval peak VRAM (GB)", lambda m: m["ft"]["peak_vram_gb"]),
    ]
    lines = [f"| metric | {a['name']} | {b['name']} |", "|---|---|---|"]
    for label, f in rows:
        lines.append(f"| {label} | {fmt(f(a))} | {fmt(f(b))} |")

    ok = {m["name"]: m["ft"]["json_valid_rate"] >= MIN_VALID for m in (a, b)}
    tie = (lo <= 0 <= hi) or abs(diff) < TIE_MARGIN
    if not any(ok.values()):
        verdict = "NO WINNER: neither model reaches the JSON-valid floor."
    elif not all(ok.values()):
        w = a if ok[a["name"]] else b
        verdict = f"WINNER: {w['name']} (the other is below the {MIN_VALID:.0%} JSON-valid floor)."
    elif tie:
        faster = a if a["ft"]["gen_tokens_per_s"] >= b["ft"]["gen_tokens_per_s"] else b
        verdict = (f"TIE on quality (diff {diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]). "
                   f"Tie-break by speed -> {faster['name']}. Re-run with a second seed before committing.")
    else:
        w = a if diff > 0 else b
        verdict = f"WINNER: {w['name']} (macro-F1 diff {diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] excludes 0)."

    md = "\n".join(lines) + f"\n\n**{verdict}**\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
