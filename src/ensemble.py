"""用 OOF 預測找 ensemble 權重，報告各單一模型、相關係數與 ensemble 分數。
若各模型都有 holdout 預測（train_* --holdout），也把同一組權重套到 holdout 上評分。
另報告「指數中性」後處理：每個時點減掉預測的指數加權平均（見 neutralize）。

    python -m src.ensemble --config configs/base.yaml --models lgb,nn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import load_config
from .cv import cv_setup
from .data import load_train
from .index_weights import load_or_estimate
from .pipeline import mae

KEY = ["stock_id", "date_id", "seconds_in_bucket"]


def load_preds(out: Path, models: list[str], prefix: str = "oof") -> pd.DataFrame:
    merged = None
    for m in models:
        d = pd.read_parquet(out / f"{prefix}_{m}.parquet").rename(columns={"pred": m})
        d = d[KEY + [c for c in ("target", "fold") if c in d.columns] + [m]]
        merged = d if merged is None else merged.merge(d[KEY + [m]], on=KEY, how="inner")
    return merged


def index_weights(cfg: dict) -> pd.Series:
    """與 pipeline.prepare 相同：只用第一折訓練日期估，對所有驗證集與 holdout 都不洩漏。"""
    dcfg = cfg["data"]
    raw = load_train(dcfg["train_path"], dcfg.get("max_dates"), dcfg.get("max_stocks"))
    folds, _ = cv_setup(raw["date_id"].to_numpy(), cfg["cv"])
    return load_or_estimate(cfg["features"].get("index_weights_path"),
                            raw[raw["date_id"].isin(folds[0].train_dates)])


def neutralize(d: pd.DataFrame, pred: np.ndarray, w: pd.Series) -> np.ndarray:
    """target 是「個股 − 指數」，同一時點依指數權重加權後約為 0；
    預測卻常帶著整體漲跌的共同成分，把它扣掉：p_i - Σ_P w_i p_i / Σ_P w_i（P 為該時點有資料的股票）。"""
    wi = d["stock_id"].map(w).fillna(0.0).to_numpy()
    keys = [d["date_id"].to_numpy(), d["seconds_in_bucket"].to_numpy()]
    g = pd.DataFrame({"wp": wi * pred, "w": wi}).groupby(keys)
    return pred - (g["wp"].transform("sum") / g["w"].transform("sum")).fillna(0.0).to_numpy()


def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """在 simplex 上最小化 MAE（權重非負、加總為 1）。"""
    k = P.shape[1]
    obj = lambda w: np.mean(np.abs(y - P @ w))
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--models", default="lgb,nn")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    models = args.models.split(",")
    d = load_preds(out, models)
    y, P = d["target"].to_numpy(), d[models].to_numpy()

    w = fit_weights(P, y)
    # 權重在全部 OOF 上找會略為樂觀；另做「用其他折找權重、套到這一折」的誠實版本
    nested = np.empty(len(d))
    for f in d["fold"].unique():
        m = d["fold"].to_numpy() == f
        nested[m] = P[m] @ fit_weights(P[~m], y[~m])

    report = {
        "single": {m: mae(y, d[m]) for m in models},
        "corr": d[models].corr().round(4).to_dict(),
        "weights": dict(zip(models, np.round(w, 4).tolist())),
        "mean_avg": mae(y, P.mean(axis=1)),
        "weighted_in_sample": mae(y, P @ w),
        "weighted_nested": mae(y, nested),
    }
    best_single = min(report["single"].values())
    report["gain_vs_best_single"] = best_single - report["weighted_nested"]
    iw = index_weights(cfg)
    report["neutral"] = {
        "single": {m: mae(y, neutralize(d, d[m].to_numpy(), iw)) for m in models},
        "weighted_nested": mae(y, neutralize(d, nested, iw)),
    }

    # holdout：只套用 OOF 找到的權重，不在 holdout 上重新找
    if all((out / f"holdout_pred_{m}.parquet").exists() for m in models):
        h = load_preds(out, models, "holdout_pred")
        yh, Ph = h["target"].to_numpy(), h[models].to_numpy()
        report["holdout"] = {
            "rows": len(h),
            "single": {m: mae(yh, h[m]) for m in models},
            "corr": h[models].corr().round(4).to_dict(),
            "mean_avg": mae(yh, Ph.mean(axis=1)),
            "weighted": mae(yh, Ph @ w),
            "neutral": {
                "single": {m: mae(yh, neutralize(h, h[m].to_numpy(), iw)) for m in models},
                "weighted": mae(yh, neutralize(h, Ph @ w, iw)),
            },
            "predict_zero": mae(yh, 0.0),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "ensemble_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
