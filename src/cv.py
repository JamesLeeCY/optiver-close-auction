"""Walk-forward（expanding window）切分，含 embargo 與最後一段 holdout。

原則：驗證集的每一天都晚於訓練集的每一天，中間再空 `embargo_days` 天。
以 date_id 切分，絕不隨機切：同一天不同股票的資料高度相關。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    fold: int
    train_dates: np.ndarray
    val_dates: np.ndarray


def split_holdout(dates: np.ndarray, holdout_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (可用於 CV 的日期, holdout 日期)。holdout 是時間上最後的那段。"""
    uniq = np.sort(np.unique(dates))
    n_hold = int(round(len(uniq) * holdout_frac))
    if n_hold == 0:
        return uniq, uniq[:0]
    return uniq[:-n_hold], uniq[-n_hold:]


def walk_forward_folds(
    dates: np.ndarray,
    n_folds: int,
    embargo_days: int,
    min_train_frac: float = 0.5,
) -> list[Fold]:
    """把日期的後半段切成 n_folds 個連續驗證區塊；每折以該區塊之前的所有日期
    （扣掉 embargo）訓練。"""
    uniq = np.sort(np.unique(dates))
    n = len(uniq)
    first_val = int(np.ceil(n * min_train_frac))
    if first_val - embargo_days <= 0:
        raise ValueError("訓練集太小：請降低 embargo_days 或提高 min_train_frac")
    bounds = np.linspace(first_val, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        v0, v1 = bounds[k], bounds[k + 1]
        if v1 <= v0:
            continue
        train_end = v0 - embargo_days
        folds.append(Fold(k, uniq[:train_end], uniq[v0:v1]))
    return folds


def cv_setup(dates: np.ndarray, cv_cfg: dict) -> tuple[list[Fold], np.ndarray]:
    cv_dates, holdout = split_holdout(dates, cv_cfg["holdout_frac"])
    folds = walk_forward_folds(cv_dates, cv_cfg["n_folds"], cv_cfg["embargo_days"],
                               cv_cfg.get("min_train_frac", 0.5))
    return folds, holdout
