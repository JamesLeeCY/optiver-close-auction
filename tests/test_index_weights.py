import numpy as np

from src.index_weights import estimate_index_weights


def test_recovers_true_weights(synth):
    df, true_w = synth
    est = estimate_index_weights(df).reindex(true_w.index)
    assert np.allclose(est.sum(), 1.0)
    assert np.abs(est - true_w).max() < 0.01


def _renormalized_panel(n_stocks=15, n_days=30, n_secs=55, seed=3):
    """每天隨機缺 1–3 檔股票（沒有任何完整時點），指數只對有資料的股票加權並重新正規化。"""
    import pandas as pd

    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n_stocks))
    frames = []
    for day in range(n_days):
        present = np.ones(n_stocks, bool)
        present[rng.choice(n_stocks, rng.integers(1, 4), replace=False)] = False
        wap = np.cumprod(1 + rng.normal(0, 5e-4, (n_secs, n_stocks)), axis=0)
        fut = np.vstack([wap[6:], np.full((6, n_stocks), np.nan)])
        r = fut / wap - 1
        wp = w * present
        idx = (r * wp).sum(axis=1) / wp.sum()
        target = (r - idx[:, None]) * 1e4
        for s in np.flatnonzero(present):
            frames.append(pd.DataFrame({"stock_id": s, "date_id": day,
                                        "seconds_in_bucket": np.arange(n_secs) * 10,
                                        "wap": wap[:, s], "target": target[:, s]}))
    return pd.concat(frames, ignore_index=True), pd.Series(w, name="weight")


def test_recovers_weights_without_complete_timestamps():
    df, true_w = _renormalized_panel()
    est = estimate_index_weights(df).reindex(true_w.index)
    assert np.allclose(est.sum(), 1.0)
    assert np.abs(est - true_w).max() < 1e-3
