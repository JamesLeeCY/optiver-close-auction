import numpy as np
import pandas as pd
import torch

from src.train_xformer import Groups, XStockNet


def _rows(n_stocks=4, n_days=2, n_steps=3, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame([(s, day, t * 10) for day in range(n_days) for t in range(n_steps) for s in range(n_stocks)],
                     columns=["stock_id", "date_id", "seconds_in_bucket"])
    d = d.drop(index=[1, 9, 10]).reset_index(drop=True)  # 有些股票在某些時點沒資料
    d = d.sample(frac=1, random_state=0).reset_index(drop=True)  # 列的順序打亂也要對得回去
    d["target"] = rng.normal(size=len(d)).astype(np.float32)
    return d, rng.normal(size=(len(d), 5)).astype(np.float32)


def test_groups_round_trip():
    d, X = _rows()
    g = Groups(d, X, n_stocks=4)
    assert g.x.shape == (6, 4, 5) and g.mask.sum() == len(d)
    assert np.allclose(g.to_rows(g.y), d["target"])
    assert np.allclose(g.x[g.grp, g.pos], X)
    assert (g.stock[g.mask] < 4).all() and (g.stock[~g.mask] == 4).all()
    # 同一組 = 同一天同一秒
    key = d["date_id"] * 1000 + d["seconds_in_bucket"]
    assert (pd.Series(g.grp).groupby(key.values).nunique() == 1).all()


def _model():
    torch.manual_seed(0)
    return XStockNet(n_num=5, n_stocks=6, emb_dim=3, d_model=8, heads=2, layers=2, ff=16, dropout=0.1).eval()


def test_padding_does_not_affect_real_stocks():
    model = _model()
    x = torch.randn(3, 6, 5)
    s = torch.tensor([[0, 1, 2, 3, 6, 6]] * 3)  # 後兩格是補位
    a = model(x, s)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(3, 2, 5) * 10
    assert torch.allclose(a[:, :4], model(x2, s)[:, :4], atol=1e-6)


def test_other_stocks_matter_and_order_does_not():
    model = _model()
    x = torch.randn(2, 6, 5)
    s = torch.arange(6).repeat(2, 1)
    a = model(x, s)
    perm = torch.randperm(6)
    assert torch.allclose(a[:, perm], model(x[:, perm], s[:, perm]), atol=1e-5)
    x2 = x.clone()
    x2[:, 5] += 3.0  # 只改第 5 檔股票，其他股票的預測也要跟著變
    assert not torch.allclose(a[:, :5], model(x2, s)[:, :5])


def test_zero_mean_over_real_stocks():
    torch.manual_seed(0)
    model = XStockNet(n_num=5, n_stocks=6, emb_dim=3, d_model=8, heads=2, layers=2, ff=16, dropout=0.1,
                      zero_mean=True).eval()
    x = torch.randn(3, 6, 5)
    s = torch.tensor([[0, 1, 2, 3, 6, 6]] * 3)
    a = model(x, s)
    assert torch.allclose(a[:, :4].mean(1), torch.zeros(3), atol=1e-6)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(3, 2, 5) * 10
    assert torch.allclose(a[:, :4], model(x2, s)[:, :4], atol=1e-6)


def test_arch_compatibility():
    from src.train_gru import _arch_runs_on
    assert _arch_runs_on("sm_60", 6, 1)          # P4000 (sm_61) 可跑 sm_60 的 cubin
    assert not _arch_runs_on("sm_70", 6, 1)
    assert not _arch_runs_on("sm_75", 6, 1)
    assert _arch_runs_on("sm_120", 12, 0) and not _arch_runs_on("sm_100", 12, 0)
    assert _arch_runs_on("compute_60", 7, 5) and not _arch_runs_on("compute_80", 7, 5)
