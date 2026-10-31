"""洩漏檢查：竄改「未來」或「其他天」的原始資料，過去的特徵不應改變。"""
import numpy as np
import pandas as pd

from src.features import NON_FEATURES, StockStats, build_features, feature_columns

INPUT_COLS = ["imbalance_size", "imbalance_buy_sell_flag", "reference_price", "matched_size",
              "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap"]


def _perturb(df: pd.DataFrame, mask: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in INPUT_COLS:
        if c == "imbalance_buy_sell_flag":
            out.loc[mask, c] = rng.choice([-1, 0, 1], mask.sum())
        else:
            out.loc[mask, c] = out.loc[mask, c] * rng.uniform(0.5, 1.5, mask.sum())
    out.loc[mask, "target"] = rng.normal(0, 100, mask.sum())
    return out


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, mask: np.ndarray):
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a.loc[mask, cols], b.loc[mask, cols], check_exact=False, rtol=1e-6)


def test_features_do_not_use_future_seconds(synth, fcfg):
    df, w = synth
    cutoff = 250
    future = (df["seconds_in_bucket"] > cutoff).to_numpy()
    a = build_features(df, fcfg, w)
    b = build_features(_perturb(df, future), fcfg, w)
    _assert_same(a, b, ~future)


def test_features_do_not_use_other_days(synth, fcfg):
    df, w = synth
    other = (df["date_id"] != 5).to_numpy()
    a = build_features(df, fcfg, w)
    b = build_features(_perturb(df, other), fcfg, w)
    _assert_same(a, b, ~other)


def test_cross_sectional_uses_same_timestamp_only(synth, fcfg):
    """把某一時點以外的資料全部改掉，該時點的橫截面特徵（非時間序列部分）不變。"""
    df, w = synth
    cfg = {**fcfg, "groups": ["basic", "auction", "time", "cross_sectional", "index"]}
    ts = (df["date_id"] == 3) & (df["seconds_in_bucket"] == 200)
    a = build_features(df, cfg, w)
    b = build_features(_perturb(df, ~ts.to_numpy()), cfg, w)
    xs_cols = [c for c in feature_columns(a) if "_xs_" in c or c.endswith("_idx")]
    xs_cols = [c for c in xs_cols if "wap_ret" not in c]  # 報酬本身就用到上一格，另由上面的測試涵蓋
    assert xs_cols
    pd.testing.assert_frame_equal(a.loc[ts, xs_cols], b.loc[ts, xs_cols], check_exact=False, rtol=1e-6)


def test_target_never_a_feature(synth, fcfg):
    df, w = synth
    cols = feature_columns(build_features(df, fcfg, w))
    assert not (set(cols) & NON_FEATURES)
    assert not any("target" in c for c in cols)


def test_stock_stats_fit_on_train_only(synth, fcfg):
    df, w = synth
    feats = build_features(df, fcfg, w)
    tr, va = feats[feats["date_id"] < 10], feats[feats["date_id"] >= 10]
    a = StockStats().fit(tr).transform(va)
    va2 = va.assign(target=va["target"] * 100)  # 改驗證集 target，不該影響
    b = StockStats().fit(tr).transform(va2)
    pd.testing.assert_frame_equal(a[StockStats.COLS], b[StockStats.COLS])


def test_wide_temporal_features_do_not_use_future_seconds(synth):
    from src.config import load_config
    from pathlib import Path

    df, w = synth
    fcfg = load_config(Path(__file__).parents[1] / "configs" / "temporal_wide.yaml")["features"]
    future = (df["seconds_in_bucket"] > 250).to_numpy()
    a = build_features(df, fcfg, w)
    b = build_features(_perturb(df, future), fcfg, w)
    assert any(c.endswith("_since_open") for c in a.columns)
    _assert_same(a, b, ~future)


def test_revealed_target_uses_only_previous_day(synth):
    """revealed 組只能用前一天的 target：改當天或更早以前的 target，當天的特徵不變。"""
    from pathlib import Path

    from src.config import load_config
    from src.features import add_revealed_target

    df, _ = synth
    day = 5
    a = add_revealed_target(df)
    rng = np.random.default_rng(0)
    for changed in (df["date_id"] == day, df["date_id"] < day - 1, df["date_id"] > day):
        b = add_revealed_target(df.assign(target=df["target"].where(~changed, rng.normal(0, 100, len(df)))))
        cols = [c for c in a.columns if c.startswith("tgt_prev_day")]
        m = (df["date_id"] == day).to_numpy()
        pd.testing.assert_frame_equal(a.loc[m, cols], b.loc[m, cols])
    # 前一天的 target 確實被用上
    prev = df[df["date_id"] == day - 1].set_index(["stock_id", "seconds_in_bucket"])["target"]
    cur = a[a["date_id"] == day].set_index(["stock_id", "seconds_in_bucket"])["tgt_prev_day"]
    assert np.allclose(cur, prev.reindex(cur.index).astype(np.float32), equal_nan=True)
    cfg = load_config(Path(__file__).parents[1] / "configs" / "revealed.yaml")["features"]
    assert "revealed" in cfg["groups"]
