"""共用流程：載資料 → 切折 → 估指數權重 → 產生特徵（有快取）。

LightGBM、MLP、ablation 全部走這裡，保證用的是同一套日期切分。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cv import Fold, cv_setup
from .data import load_train
from .features import StockStats, build_features, feature_columns
from .index_weights import load_or_estimate


@dataclass
class Prepared:
    df: pd.DataFrame
    folds: list[Fold]
    holdout_dates: np.ndarray
    features: list[str]
    index_weights: pd.Series


def _cache_key(cfg: dict) -> str:
    payload = json.dumps({"data": cfg["data"], "cv": cfg["cv"], "features": cfg["features"]},
                         sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:10]


def prepare(cfg: dict, use_cache: bool = True) -> Prepared:
    dcfg = cfg["data"]
    raw = load_train(dcfg["train_path"], dcfg.get("max_dates"), dcfg.get("max_stocks"))
    folds, holdout = cv_setup(raw["date_id"].to_numpy(), cfg["cv"])

    # 指數權重只用「第一折訓練日期」估：它是所有折訓練集的子集，對任何驗證集都不洩漏
    first_train = raw[raw["date_id"].isin(folds[0].train_dates)]
    weights = load_or_estimate(cfg["features"].get("index_weights_path"), first_train)

    cache = Path(dcfg["train_path"]).parent / "cache" / f"features_{_cache_key(cfg)}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = build_features(raw, cfg["features"], weights)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
    return Prepared(df, folds, holdout, feature_columns(df), weights)


def fold_frames(p: Prepared, train_dates, val_dates, use_stock_stats: bool):
    """切出一折的 train / val，個股統計量只從該折訓練資料 fit。"""
    tr = p.df[p.df["date_id"].isin(train_dates)]
    va = p.df[p.df["date_id"].isin(val_dates)]
    feats = list(p.features)
    if use_stock_stats:
        ss = StockStats().fit(tr)
        tr, va = ss.transform(tr), ss.transform(va)
        feats += [c for c in StockStats.COLS if c in tr.columns]
    return tr, va, feats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
