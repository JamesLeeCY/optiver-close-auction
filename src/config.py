"""讀取 yaml 設定，支援 `inherit:` 以覆寫 base 設定。"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg
