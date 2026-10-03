"""Phase 3 table: quality vs size for each model variant, all scored on the same test set.

Usage: python -m src.eval.quant_table --variant NAME RESULTS_DIR MODEL_PATH [--variant ...]
RESULTS_DIR holds finetuned_test.json + finetuned_test_preds.jsonl; MODEL_PATH is the file/dir whose size to report.
The first variant is the reference (fp16): "agreement" is the share of test examples where a variant gives the
same intent as the reference, and "delta" is the accuracy change vs the reference.
"""
import argparse
import json
from pathlib import Path

from src.eval.compare import intent_stats


def size_gb(path: str) -> float:
    # MODEL_PATH may be a real file/dir, a number (size in GB measured elsewhere), or "n/a" if unknown.
    try:
        return float(path)
    except ValueError:
        pass
    if path.lower() in ("n/a", "na", "?") or not Path(path).exists():
        return float("nan")
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs=3, action="append", metavar=("NAME", "RESULTS_DIR", "MODEL_PATH"), required=True)
    ap.add_argument("--out", default="results/phase3/quant_table.md")
    args = ap.parse_args()

    rows, ref = [], None
    for name, rdir, mpath in args.variant:
        m = json.loads((Path(rdir) / "finetuned_test.json").read_text())
        preds = [json.loads(l) for l in open(Path(rdir) / "finetuned_test_preds.jsonl", encoding="utf-8")]
        st = intent_stats(preds)
        if ref is None:
            ref = {"texts": [p["text"] for p in preds], "pred": [p["pred"] for p in preds], "acc": st["acc"]}
        assert [p["text"] for p in preds] == ref["texts"], f"{name}: evaluated on different examples"
        agree = sum(a == b for a, b in zip(ref["pred"], (p["pred"] for p in preds))) / len(preds)
        rows.append((name, size_gb(mpath), st["acc"], st["acc"] - ref["acc"], st["f1"], st["invented"],
                     m["json_valid_rate"], m["urgency_accuracy"], m["needs_human_accuracy"], agree))

    head = ("| variant | size (GB) | intent acc | delta vs fp16 | macro-F1 | invented labels | valid JSON | "
            "urgency | needs_human | same answer as fp16 |")
    lines = [head, "|" + "---|" * 10]
    for n, sz, acc, d, f1, inv, js, urg, hum, ag in rows:
        size_txt = "n/a" if sz != sz else f"{sz:.2f}"
        lines.append(f"| {n} | {size_txt} | {acc:.4f} | {d:+.4f} | {f1:.4f} | {inv:.2%} | {js:.2%} | "
                     f"{urg:.4f} | {hum:.4f} | {ag:.2%} |")
    md = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
