"""載入比賽資料，數值欄位一律 float32 以省記憶體（524 萬列 × 多欄在 8GB 筆電上要省著用）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["stock_id", "date_id", "seconds_in_bucket"]
AUCTION_COLS = ["imbalance_size", "imbalance_buy_sell_flag", "reference_price",
                "matched_size", "far_price", "near_price"]
BOOK_COLS = ["bid_price", "bid_size", "ask_price", "ask_size", "wap"]
TARGET = "target"

DTYPES = {
    "stock_id": np.int16,
    "date_id": np.int16,
    "seconds_in_bucket": np.int16,
    "imbalance_buy_sell_flag": np.int8,
    **{c: np.float32 for c in AUCTION_COLS + BOOK_COLS if c != "imbalance_buy_sell_flag"},
    TARGET: np.float32,
}


def load_train(path: str, max_dates: int | None = None, max_stocks: int | None = None) -> pd.DataFrame:
    usecols = KEYS + AUCTION_COLS + BOOK_COLS + [TARGET]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=DTYPES)
    if max_dates is not None:
        df = df[df["date_id"] < max_dates]
    if max_stocks is not None:
        df = df[df["stock_id"] < max_stocks]
    # target 有少量缺值（該股當天資料不足 60 秒），無法評分，直接移除
    df = df[df[TARGET].notna()]
    return df.sort_values(["date_id", "seconds_in_bucket", "stock_id"]).reset_index(drop=True)
