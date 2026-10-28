"""產生與比賽 train.csv 同 schema 的合成資料，用來在沒有真實資料時跑通 pipeline 與 CI。

target 依比賽定義計算：(個股 WAP 60 秒報酬 - 加權指數 60 秒報酬) × 1e4，
並讓「帶方向的競價失衡」對未來報酬有真實的預測力，所以模型應該學得到東西。
同時輸出真實權重 data/synthetic_weights.csv，供測試檢查 index_weights 估計是否正確。

    python scripts/make_synthetic.py --stocks 20 --days 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_stocks: int, n_days: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    T = 55  # 0..540 秒，每 10 秒一格
    weights = rng.dirichlet(np.ones(n_stocks))
    vol = rng.uniform(5e-4, 2e-3, n_stocks)
    frames = []
    for d in range(n_days):
        mkt = rng.normal(0, 4e-4, T)
        imb = np.zeros((T, n_stocks))
        imb[0] = rng.normal(0, 1, n_stocks)
        for t in range(1, T):  # 失衡有持續性
            imb[t] = 0.9 * imb[t - 1] + rng.normal(0, 0.4, n_stocks)
        signal = 3e-4 * np.tanh(imb)  # 失衡越大，下一格越往同方向走
        ret = mkt[:, None] + signal + rng.normal(0, 1, (T, n_stocks)) * vol
        wap = np.cumprod(1 + ret, axis=0)
        wap = wap / wap[0]

        sec = np.arange(T) * 10
        matched = rng.lognormal(15, 1, n_stocks)[None, :] * (1 + 0.02 * np.arange(T))[:, None]
        imb_size = np.abs(imb) * matched * 0.05
        flag = np.sign(imb).astype(int)
        spread = vol * rng.uniform(0.5, 1.5, (T, n_stocks))
        bid, ask = wap * (1 - spread / 2), wap * (1 + spread / 2)
        bsz = rng.lognormal(10, 1, (T, n_stocks)) * (1 + 0.3 * np.tanh(imb))
        asz = rng.lognormal(10, 1, (T, n_stocks)) * (1 - 0.3 * np.tanh(imb))
        ref = wap * (1 + 2e-4 * np.tanh(imb))
        far = np.where(sec[:, None] >= 300, ref * (1 + rng.normal(0, 1e-3, (T, n_stocks))), np.nan)
        near = np.where(sec[:, None] >= 300, ref * (1 + rng.normal(0, 3e-4, (T, n_stocks))), np.nan)

        fut = np.full_like(wap, np.nan)
        fut[:-6] = wap[6:] / wap[:-6] - 1
        idx_ret = fut @ weights
        target = (fut - idx_ret[:, None]) * 1e4

        frames.append(pd.DataFrame({
            "stock_id": np.tile(np.arange(n_stocks), T),
            "date_id": d,
            "seconds_in_bucket": np.repeat(sec, n_stocks),
            "imbalance_size": imb_size.ravel(),
            "imbalance_buy_sell_flag": flag.ravel(),
            "reference_price": ref.ravel(),
            "matched_size": matched.ravel(),
            "far_price": far.ravel(),
            "near_price": near.ravel(),
            "bid_price": bid.ravel(),
            "bid_size": bsz.ravel(),
            "ask_price": ask.ravel(),
            "ask_size": asz.ravel(),
            "wap": wap.ravel(),
            "target": target.ravel(),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["time_id"] = df["date_id"] * T + df["seconds_in_bucket"] // 10
    df["row_id"] = (df["date_id"].astype(str) + "_" + df["seconds_in_bucket"].astype(str)
                    + "_" + df["stock_id"].astype(str))
    return df, pd.Series(weights, index=pd.Index(range(n_stocks), name="stock_id"), name="weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=20)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    df, w = make(args.stocks, args.days)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "synthetic_train.csv", index=False)
    w.to_csv(out / "synthetic_weights.csv")
    print(f"wrote {len(df):,} rows to {out / 'synthetic_train.csv'}")


if __name__ == "__main__":
    main()
