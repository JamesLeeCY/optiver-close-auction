"""所有特徵。每個 add_* 都是純函式：輸入 DataFrame，回傳加了新欄位的新 DataFrame。

防洩漏規則（tests/test_leakage.py 會檢查）：
- 時間特徵只用同一 (stock_id, date_id) 內 seconds_in_bucket 更早（含當下）的資料
- 橫截面特徵只用同一 (date_id, seconds_in_bucket) 的資料，不用整天統計量
- 個股統計量（StockStats）只從訓練折 fit

輸入需依 (date_id, seconds_in_bucket, stock_id) 排序，groupby 後的 shift 才是「上一格」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
SEQ = ["stock_id", "date_id"]          # 同股同日的時間序列
XS = ["date_id", "seconds_in_bucket"]  # 同一時點的橫截面
NON_FEATURES = {"target", "date_id", "row_id", "time_id"}


def _f32(s) -> pd.Series:
    return s.astype(np.float32)


# ---------- 價差與失衡：掛單量的不對稱反映短期買賣壓力 ----------
def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bid, ask = df["bid_price"], df["ask_price"]
    bs, as_ = df["bid_size"], df["ask_size"]
    df["mid"] = _f32((bid + ask) / 2)
    df["spread"] = _f32((ask - bid) / (ask + bid + EPS))
    df["book_imb"] = _f32((bs - as_) / (bs + as_ + EPS))
    df["liquidity"] = _f32(np.log1p(bs + as_))
    df["wap_mid_gap"] = _f32(df["wap"] / (df["mid"] + EPS) - 1)
    return df


# ---------- 競價失衡：未配對量會推動最終收盤價 ----------
def add_auction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flag = df["imbalance_buy_sell_flag"].astype(np.float32)
    df["imb_signed"] = _f32(df["imbalance_size"] * flag)
    df["imb_ratio"] = _f32(df["imbalance_size"] / (df["matched_size"] + EPS))
    df["imb_ratio_signed"] = _f32(df["imb_ratio"] * flag)
    df["imb_vs_book"] = _f32(df["imb_signed"] / (df["bid_size"] + df["ask_size"] + EPS))
    # 價格之間的距離：競價參考價與連續市場價的落差，可能在收盤前收斂
    df["ref_wap_gap"] = _f32(df["reference_price"] / df["wap"] - 1)
    df["ref_mid_gap"] = _f32(df["reference_price"] / (df["mid"] if "mid" in df else df["wap"]) - 1)
    df["far_near_gap"] = _f32(df["far_price"] - df["near_price"])
    df["near_ref_gap"] = _f32(df["near_price"] - df["reference_price"])
    df["far_ref_gap"] = _f32(df["far_price"] - df["reference_price"])
    return df


# ---------- 時段：300 秒後 far / near price 才出現，市場行為不同 ----------
def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["after_300"] = (df["seconds_in_bucket"] >= 300).astype(np.int8)
    df["time_to_close"] = (540 - df["seconds_in_bucket"]).astype(np.int16)
    return df


# ---------- 時間動態：失衡在擴大還是收斂（只往前看） ----------
def _rolling_mean_past(df: pd.DataFrame, col: str, w: int) -> pd.Series:
    """同股同日內，含當下在內往前 w 格的平均。用 cumsum 實作，比 groupby.rolling 快很多。"""
    keys = [df[k] for k in SEQ]
    x = df[col].astype(np.float64)
    cs = x.fillna(0).groupby(keys, sort=False).cumsum()
    cnt = x.notna().astype(np.float64).groupby(keys, sort=False).cumsum()
    cs_prev = cs.groupby(keys, sort=False).shift(w).fillna(0)
    cnt_prev = cnt.groupby(keys, sort=False).shift(w).fillna(0)
    return _f32((cs - cs_prev) / (cnt - cnt_prev).replace(0, np.nan))


def add_returns(df: pd.DataFrame, horizons=(1, 3, 6)) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(SEQ, sort=False)["wap"]
    for h in horizons:
        df[f"wap_ret_{h}"] = _f32(df["wap"] / g.shift(h) - 1)
    return df


def add_temporal(df: pd.DataFrame, cols, lags=(1, 2, 3, 6), windows=(3, 6),
                 ret_horizons=(1, 3, 6), open_cols=()) -> pd.DataFrame:
    df = add_returns(df, ret_horizons)
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(SEQ, sort=False)
    new = {}
    # 相對當天第一格的變化：只用該股該日最早一列，不用 transform("first")（它會跳過缺值、拿到之後的值）
    is_open = (g.cumcount() == 0).to_numpy()
    for c in [c for c in open_cols if c in df.columns]:
        opening = df[c].where(is_open).groupby([df[k] for k in SEQ], sort=False).transform("max")
        new[f"{c}_since_open"] = _f32(df[c] - opening)
    for c in cols:
        for lag in lags:
            prev = g[c].shift(lag)
            new[f"{c}_lag{lag}"] = _f32(prev)
            new[f"{c}_diff{lag}"] = _f32(df[c] - prev)
        for w in windows:
            new[f"{c}_rmean{w}"] = _rolling_mean_past(df, c, w)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ---------- 橫截面：target 是相對指數，相對特徵直接對應預測目標 ----------
def add_cross_sectional(df: pd.DataFrame, cols) -> pd.DataFrame:
    if "wap_ret_1" not in df.columns:
        df = add_returns(df)
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(XS, sort=False)
    new = {}
    for c in cols:
        new[f"{c}_xs_demean"] = _f32(df[c] - g[c].transform("mean"))
        new[f"{c}_xs_rank"] = _f32(g[c].rank(pct=True))
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ---------- 指數相對：用（估計的）指數權重重建同時點的指數報酬 ----------
def add_index_relative(df: pd.DataFrame, weights: pd.Series | None) -> pd.DataFrame:
    if "wap_ret_1" not in df.columns:
        df = add_returns(df)
    df = df.copy()
    if weights is None:
        w = pd.Series(1.0, index=df["stock_id"].unique())
    else:
        w = weights
    df["_w"] = df["stock_id"].map(w).fillna(0).astype(np.float64)
    g = df.groupby(XS, sort=False)
    wsum = g["_w"].transform("sum").replace(0, np.nan)
    for c in [c for c in df.columns if c.startswith("wap_ret_")]:
        idx = (df[c] * df["_w"]).groupby([df[k] for k in XS], sort=False).transform("sum") / wsum
        df[f"idx_{c}"] = _f32(idx)
        df[f"rel_{c}"] = _f32(df[c] - idx)
    for c in ["imb_ratio_signed", "book_imb"]:
        if c in df.columns:
            m = (df[c] * df["_w"]).groupby([df[k] for k in XS], sort=False).transform("sum") / wsum
            df[f"{c}_idx"] = _f32(m)
    df["index_weight"] = _f32(df["_w"])
    return df.drop(columns="_w")


# ---------- 個股層級：不同股票的雜訊尺度不同（只從訓練折 fit） ----------
class StockStats:
    COLS = ["stock_tgt_std", "stock_tgt_mad", "stock_imb_med", "stock_spread_med"]

    def fit(self, train: pd.DataFrame) -> "StockStats":
        g = train.groupby("stock_id")
        stats = pd.DataFrame({
            "stock_tgt_std": g["target"].std(),
            "stock_tgt_mad": g["target"].apply(lambda s: (s - s.median()).abs().median()),
        })
        if "imb_ratio" in train:
            stats["stock_imb_med"] = g["imb_ratio"].median()
        if "spread" in train:
            stats["stock_spread_med"] = g["spread"].median()
        self.stats_ = stats.astype(np.float32)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=[c for c in self.COLS if c in df.columns])
        return out.merge(self.stats_, left_on="stock_id", right_index=True, how="left")


def build_features(df: pd.DataFrame, fcfg: dict, index_weights: pd.Series | None = None) -> pd.DataFrame:
    """依 config 的 groups 產生特徵（不含需要 fit 的 stock 組，那組在訓練迴圈裡每折各自處理）。"""
    groups = fcfg["groups"]
    if "basic" in groups:
        df = add_basic(df)
    if "auction" in groups:
        df = add_auction(df)
    if "time" in groups:
        df = add_time(df)
    if "temporal" in groups:
        t = fcfg.get("temporal", {})
        df = add_temporal(df, t.get("cols", []), t.get("lags", (1, 2, 3, 6)), t.get("windows", (3, 6)),
                          t.get("ret_horizons", (1, 3, 6)), t.get("open_cols", ()))
    if "cross_sectional" in groups:
        df = add_cross_sectional(df, fcfg.get("cross_sectional", {}).get("cols", []))
    if "index" in groups:
        df = add_index_relative(df, index_weights)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


# 特徵組 → 欄位，供 ablation 使用
def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    raw = {"stock_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
           "reference_price", "matched_size", "far_price", "near_price",
           "bid_price", "bid_size", "ask_price", "ask_size", "wap"}
    rules = {
        "basic": lambda c: c in {"mid", "spread", "book_imb", "liquidity", "wap_mid_gap"},
        "auction": lambda c: c in {"imb_signed", "imb_ratio", "imb_ratio_signed", "imb_vs_book",
                                   "ref_wap_gap", "ref_mid_gap", "far_near_gap",
                                   "near_ref_gap", "far_ref_gap"},
        "time": lambda c: c in {"after_300", "time_to_close"},
        "temporal": lambda c: any(s in c for s in ("_lag", "_diff", "_rmean", "_since_open")) or
                              (c.startswith("wap_ret_")),
        "cross_sectional": lambda c: "_xs_" in c,
        "index": lambda c: c.startswith(("idx_", "rel_")) or c.endswith("_idx") or c == "index_weight",
        "stock": lambda c: c in StockStats.COLS,
    }
    out = {"raw": [c for c in cols if c in raw]}
    for name, rule in rules.items():
        out[name] = [c for c in cols if rule(c) and c not in raw]
    return out
