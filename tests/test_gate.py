"""Proof that the gate is not decorative: good predictions pass, degraded predictions fail.

Runs in CI on every push (no model needed). Uses the real Phase 4 predictions of the Q4_K_M model when present,
and builds degraded versions by damaging them in ways that mimic real failures.
"""
import json
import random
from pathlib import Path

import pytest
import yaml

from src.eval.gate import check
from src.eval.metrics import score_rows

ROOT = Path(__file__).resolve().parents[1]
THR = yaml.safe_load(open(ROOT / "configs/gate.yaml"))
SLICE = [json.loads(l) for l in open(ROOT / "data/gate_slice.jsonl", encoding="utf-8")]
GOLDS = [r["label"] for r in SLICE]


def perfect_outputs():
    return [json.dumps(g) for g in GOLDS]


def test_perfect_model_passes():
    m, _ = score_rows(GOLDS, perfect_outputs())
    assert check(m, THR)[1]


def test_untuned_model_prose_fails():
    # a base model answering in free text instead of JSON
    m, _ = score_rows(GOLDS, ["The customer seems to be asking about their card."] * len(GOLDS))
    assert not check(m, THR)[1]


def test_wrong_labels_fail():
    # a model that is valid JSON but picks a wrong intent 25% of the time
    rng = random.Random(0)
    labels = sorted({g["intent"] for g in GOLDS})
    outs = []
    for g in GOLDS:
        g = dict(g)
        if rng.random() < 0.25:
            g["intent"] = rng.choice(labels)
        outs.append(json.dumps(g))
    m, _ = score_rows(GOLDS, outs)
    assert not check(m, THR)[1]


def test_invented_labels_fail():
    outs = [json.dumps({**g, "intent": "made_up_label"}) if i % 10 == 0 else json.dumps(g) for i, g in enumerate(GOLDS)]
    m, _ = score_rows(GOLDS, outs)
    assert not check(m, THR)[1]


def test_missing_metric_fails():
    assert not check({"json_valid_rate": 1.0}, THR)[1]


def test_real_q4km_predictions_pass():
    p = ROOT / "phase4_results/gguf-q4km/finetuned_test_preds.jsonl"
    if not p.exists():
        pytest.skip("Phase 4 predictions not in the repo")
    preds = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        preds.setdefault((r["text"], r["gold"]), r["raw"])
    outs = [preds[(r["prompt"][1]["content"], r["label"]["intent"])] for r in SLICE]
    m, _ = score_rows(GOLDS, outs)
    assert check(m, THR)[1], m
