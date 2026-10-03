"""Metric computation shared by the HTTP evaluator and the local HF evaluator."""
from sklearn.metrics import f1_score

from src.schema import parse_output

INVALID = "__invalid__"


def score_rows(golds: list[dict], texts: list[str]):
    """Return (metrics dict, per-example records). `golds` are the rows' `label` dicts."""
    y_true, y_pred, recs = [], [], []
    valid = urg_ok = hum_ok = 0
    for gold, text in zip(golds, texts):
        pred = parse_output(text)
        y_true.append(gold["intent"])
        if pred is None:
            y_pred.append(INVALID)
            recs.append({"gold": gold["intent"], "pred": INVALID, "raw": text})
            continue
        valid += 1
        y_pred.append(pred["intent"])
        urg_ok += pred["urgency"] == gold["urgency"]
        hum_ok += pred["needs_human"] == gold["needs_human"]
        recs.append({"gold": gold["intent"], "pred": pred["intent"], "raw": text})
    n = len(golds)
    real = sorted(set(y_true))
    # Macro-F1 over the real labels only. sklearn's default would also average in every
    # *invented* label as an extra class with F1=0, which penalises noise, not the model.
    invented = sum(1 for p in y_pred if p != INVALID and p not in set(real))
    metrics = {
        "n": n,
        "json_valid_rate": valid / n,
        "intent_accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / n,
        "intent_macro_f1": f1_score(y_true, y_pred, labels=real, average="macro", zero_division=0),
        "invented_label_rate": invented / n,
        "urgency_accuracy": urg_ok / n,
        "needs_human_accuracy": hum_ok / n,
    }
    return metrics, recs
