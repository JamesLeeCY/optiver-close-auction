import numpy as np
import pandas as pd
import torch

from src.train_gru import GRUNet, Sequences


def _rows(n_stocks=3, n_days=2, n_steps=5, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame([(s, day, t * 10) for day in range(n_days) for t in range(n_steps) for s in range(n_stocks)],
                     columns=["stock_id", "date_id", "seconds_in_bucket"])
    d = d.drop(index=[4, 11]).reset_index(drop=True)  # 有些時點缺資料
    d["target"] = rng.normal(size=len(d)).astype(np.float32)
    return d, rng.normal(size=(len(d), 4)).astype(np.float32)


def test_sequences_round_trip():
    d, X = _rows()
    s = Sequences(d, X, n_steps=5)
    assert s.x.shape == (6, 5, 4) and s.mask.sum() == len(d)
    assert np.allclose(s.to_rows(s.y), d["target"])
    assert np.allclose(s.x[s.seq, s.step], X)
    assert (s.stock[s.seq] == d["stock_id"]).all()


def test_gru_output_does_not_use_future_steps():
    torch.manual_seed(0)
    model = GRUNet(n_num=4, n_stocks=3, emb_dim=2, hidden=8, layers=2, dropout=0.1).eval()
    x = torch.randn(5, 10, 4)
    s = torch.tensor([0, 1, 2, 0, 1])
    a = model(x, s)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(5, 4, 4) * 10  # 竄改第 6 格之後
    b = model(x2, s)
    assert torch.allclose(a[:, :6], b[:, :6])
    assert not torch.allclose(a[:, 6:], b[:, 6:])
