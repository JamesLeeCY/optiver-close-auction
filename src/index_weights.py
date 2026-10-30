"""從資料反推合成指數的權重。

target_i = (r_i - r_index) * 1e4，其中 r_i 是個股 WAP 未來 60 秒（6 格）報酬。
所以 r_i - target_i / 1e4 在同一時點對所有股票都等於 r_index。

指數只對「該時點有資料的股票」加權並重新正規化：r_index = Σ_P w_i r_i / Σ_P w_i（P 為有資料的股票）。
移項得 Σ_P w_i (r_i - r_index) = 0，對 w 是線性的，因此可以用上所有時點（不只全部股票都在的時點），
再加一列「權重總和 = 1」的限制，以非負最小平方（NNLS）求解。

注意：這裡用到未來 WAP 與 target，只能在「訓練日期」上估計，且只當成一組固定常數
（指數定義），不當特徵。train 腳本只用第一折的訓練日期來估，對所有折都不洩漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HORIZON = 6  # 60 秒 / 10 秒一格


def estimate_index_weights(df: pd.DataFrame) -> pd.Series:
    d = df[["stock_id", "date_id", "seconds_in_bucket", "wap", "target"]].copy()
    d = d.sort_values(["stock_id", "date_id", "seconds_in_bucket"])
    fut = d.groupby(["stock_id", "date_id"])["wap"].shift(-HORIZON)
    d["r"] = fut / d["wap"] - 1.0
    d["idx"] = d["r"] - d["target"] / 1e4
    d = d.dropna(subset=["r", "idx"])

    R = d.pivot_table(index=["date_id", "seconds_in_bucket"], columns="stock_id", values="r")
    y = d.groupby(["date_id", "seconds_in_bucket"])["idx"].median().reindex(R.index)
    if len(R) < R.shape[1]:
        raise ValueError(f"可用時點 {len(R)} 少於股票數 {R.shape[1]}，無法估權重")
    # 每列：Σ_P w_i (r_i - r_index) = 0；缺值的股票不在 P 裡，係數為 0
    A = np.nan_to_num(R.to_numpy(np.float64) - y.to_numpy(np.float64)[:, None])
    # 最後一列以大權重逼近「Σ w = 1」，避免全部為 0 的解
    lam = 1e3 * np.sqrt((A ** 2).sum())
    A = np.vstack([A, np.full(A.shape[1], lam)])
    b = np.zeros(len(A))
    b[-1] = lam
    w, _ = nnls(A, b)
    w = w / w.sum()
    return pd.Series(w, index=R.columns, name="weight")


def load_or_estimate(path: str | None, train_df: pd.DataFrame) -> pd.Series:
    if path:
        s = pd.read_csv(path).set_index("stock_id")["weight"]
        return s / s.sum()
    return estimate_index_weights(train_df)
