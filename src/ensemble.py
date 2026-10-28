"""用 OOF 預測找 ensemble 權重，報告各單一模型、相關係數與 ensemble 分數。

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
from .pipeline import mae

KEY = ["stock_id", "date_id", "seconds_in_bucket"]


def load_oofs(out: Path, models: list[str]) -> pd.DataFrame:
    merged = None
    for m in models:
        d = pd.read_parquet(out / f"oof_{m}.parquet").rename(columns={"pred": m})
        d = d[KEY + ["target", "fold", m]]
        merged = d if merged is None else merged.merge(d[KEY + [m]], on=KEY, how="inner")
    return merged


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
    out = Path(load_config(args.config)["output_dir"])
    models = args.models.split(",")
    d = load_oofs(out, models)
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
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "ensemble_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
