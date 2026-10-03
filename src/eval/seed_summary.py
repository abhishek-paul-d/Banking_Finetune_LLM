"""Does the pilot model comparison survive a second training seed?

Pre-registered rule (written before the seed-1 run):
  d_s = macro-F1(model A) - macro-F1(model B) on the full test set, for each seed s, with a paired-bootstrap 95% CI.
  TIE CONFIRMED  : in every seed |d_s| < 0.01 or the CI includes 0.
  A (or B) AHEAD : in every seed the CI excludes 0 in the same direction -> the model choice must be reopened if it is not the one we picked.
  MIXED          : anything else (the seeds disagree) -> treat as a tie and say so.
Also reports how much the SAME model moves between seeds, which is the noise level the comparison has to beat.

Usage: python -m src.eval.seed_summary --models qwen3_5_4b gemma4_e4b --roots outputs/pilot outputs/pilot --suffixes "" _seed1
"""
import argparse
import json
from pathlib import Path

from src.eval.compare import TIE_MARGIN, intent_stats, macro_f1, paired_bootstrap


def load_preds(root, name):
    return [json.loads(l) for l in open(Path(root) / name / "finetuned_test_preds.jsonl", encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, required=True, help="base names of model A and model B")
    ap.add_argument("--roots", nargs="+", required=True, help="one results root per seed")
    ap.add_argument("--suffixes", nargs="+", required=True, help="folder-name suffix per seed, e.g. '' _seed1")
    ap.add_argument("--out", default="results/pilot/seed_summary.md")
    args = ap.parse_args()
    assert len(args.roots) == len(args.suffixes)
    a, b = args.models
    n_seeds = len(args.roots)

    P = {m: [load_preds(r, m + s) for r, s in zip(args.roots, args.suffixes)] for m in (a, b)}
    texts = [p["text"] for p in P[a][0]]
    assert all([p["text"] for p in preds] == texts for m in P for preds in P[m]), "different eval rows"

    lines = ["| model | seed | intent acc | macro-F1 | invented labels |", "|---|---|---|---|---|"]
    stats = {m: [intent_stats(p) for p in P[m]] for m in P}
    for m in (a, b):
        for i, st in enumerate(stats[m]):
            lines.append(f"| {m} | {i} | {st['acc']:.4f} | {st['f1']:.4f} | {st['invented']:.2%} |")
        f1s = [st["f1"] for st in stats[m]]
        lines.append(f"| **{m}** | mean (range) | {sum(s['acc'] for s in stats[m]) / n_seeds:.4f} | "
                     f"{sum(f1s) / n_seeds:.4f} ({min(f1s):.4f} to {max(f1s):.4f}) | |")

    lines += ["", f"Difference in macro-F1, {a} minus {b} (paired bootstrap 95% CI):", "",
              "| seed | diff | 95% CI | verdict for this seed |", "|---|---|---|---|"]
    rows = []
    everyone = range(len(texts))
    for i in range(n_seeds):
        d = macro_f1(P[a][i], everyone) - macro_f1(P[b][i], everyone)
        lo, hi = paired_bootstrap(P[a][i], P[b][i])
        tie = (lo <= 0 <= hi) or abs(d) < TIE_MARGIN
        rows.append((d, lo, hi, tie))
        v = "tie" if tie else (f"{a} ahead" if d > 0 else f"{b} ahead")
        lines.append(f"| {i} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {v} |")
    mean_d = sum(r[0] for r in rows) / n_seeds
    lines.append(f"| mean | {mean_d:+.4f} | | |")

    if all(r[3] for r in rows):
        verdict = f"TIE CONFIRMED across {n_seeds} seeds (mean diff {mean_d:+.4f}). The choice rests on speed and memory, not quality."
    elif all(not r[3] and r[0] > 0 for r in rows):
        verdict = f"{a} AHEAD in every seed (mean diff {mean_d:+.4f})."
    elif all(not r[3] and r[0] < 0 for r in rows):
        verdict = f"{b} AHEAD in every seed (mean diff {mean_d:+.4f})."
    else:
        verdict = f"MIXED: the seeds disagree (mean diff {mean_d:+.4f}); treat as a tie."

    # how far the same model moves between seeds, as a fraction of examples whose answer changed
    for m in (a, b):
        ch = sum(x["pred"] != y["pred"] for x, y in zip(P[m][0], P[m][1])) / len(texts)
        lines.append(f"\n{m}: {ch:.1%} of test answers differ between seed 0 and seed 1.")
    md = "\n".join(lines) + f"\n\n**{verdict}**\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
