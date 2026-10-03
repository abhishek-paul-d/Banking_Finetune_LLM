"""YAML config loading with a one-level `extends:` for shared hyperparameters."""
from pathlib import Path

import yaml


def deep_update(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(load_cfg(parent), cfg)
    return cfg
