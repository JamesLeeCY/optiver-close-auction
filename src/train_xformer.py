"""跨股票 Transformer：同一個時點（date, seconds）的所有股票當成一組，用 self-attention 互相看。

target 是「個股相對指數」的報酬，同一時刻其他股票的狀態很重要；LightGBM / MLP / GRU 都一次只看
一檔股票，只能靠手工的橫斷面特徵補。這裡每個 token 是一檔股票在該時點的特徵 + stock embedding，
不加位置編碼（股票之間沒有順序）。同一時點的所有股票在比賽 API 中同時給出，不會用到未來資訊。
沒有時間維度，所以保留 temporal 特徵組。訓練、early stopping 與 holdout 流程沿用 train_gru。

    python -m src.train_xformer --config configs/xformer.yaml
    python -m src.train_xformer --config configs/xformer.yaml --holdout-epochs 8
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .pipeline import mae
from .train_gru import STEP, _fit_seed, cli, gru_features
from .train_nn import Preprocessor, _numeric, seeds_of


class XStockNet(nn.Module):
    """stock id == n_stocks 的位置是補位，attention 不看它、也不算進 loss。"""

    def __init__(self, n_num: int, n_stocks: int, emb_dim: int, d_model: int, heads: int,
                 layers: int, ff: int, dropout: float, zero_mean: bool = False):
        super().__init__()
        self.pad = n_stocks
        self.zero_mean = zero_mean
        self.emb = nn.Embedding(n_stocks + 1, emb_dim, padding_idx=n_stocks)
        self.inp = nn.Sequential(nn.Linear(n_num + emb_dim, d_model), nn.SiLU())
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    def forward(self, x, s):
        """x: [B, N, F]，s: [B, N] → [B, N]"""
        h = self.inp(torch.cat([x, self.emb(s)], dim=2))
        pad = s == self.pad
        o = self.enc(h, src_key_padding_mask=pad)
        out = self.head(torch.cat([o, h], dim=2)).squeeze(-1)
        if self.zero_mean:  # 每個時點減掉真實股票的平均，讓輸出本身就是零均值（第一名的做法）
            real = (~pad).float()
            out = out - (out * real).sum(1, keepdim=True) / real.sum(1, keepdim=True).clamp(min=1)
        return out


class Groups:
    """把列排成 [時點, 股票, 特徵]，第 i 個 token 就是 stock_id i；當時沒有資料的股票補位。"""

    def __init__(self, df: pd.DataFrame, X: np.ndarray, n_stocks: int):
        step = df["seconds_in_bucket"].to_numpy(np.int64) // STEP
        key = df["date_id"].to_numpy(np.int64) * 1000 + step
        uniq, self.grp = np.unique(key, return_inverse=True)
        self.pos = df["stock_id"].to_numpy(np.int64)
        self.x = np.zeros((len(uniq), n_stocks, X.shape[1]), dtype=np.float32)
        self.x[self.grp, self.pos] = X
        self.y = np.zeros((len(uniq), n_stocks), dtype=np.float32)
        self.y[self.grp, self.pos] = df["target"].to_numpy(np.float32)
        self.mask = np.zeros((len(uniq), n_stocks), dtype=bool)
        self.mask[self.grp, self.pos] = True
        self.stock = np.where(self.mask, np.arange(n_stocks), n_stocks).astype(np.int64)

    def to_rows(self, p: np.ndarray) -> np.ndarray:
        return p[self.grp, self.pos]


def train_fold(tr, va, feats, xcfg, n_stocks, n_steps, epochs=None, verbose=True):
    """介面同 train_gru.train_fold；n_steps 用不到，只為了讓 cli 共用。"""
    fixed = epochs is not None
    num = [c for c in gru_features(feats, xcfg) if c != "stock_id"]
    Xtr_raw = _numeric(tr, num)
    pre = Preprocessor(xcfg["scaler"], xcfg["seed"]).fit(Xtr_raw)
    gtr = Groups(tr, pre.transform(Xtr_raw), n_stocks)
    del Xtr_raw
    gva = Groups(va, pre.transform(_numeric(va, num)), n_stocks)
    yva = None if fixed else va["target"].to_numpy(np.float32)
    if verbose:
        print(f" {len(num)} features, {len(gtr.x)} train / {len(gva.x)} val timestamps")

    def build(n_num):
        return XStockNet(n_num, n_stocks, xcfg["stock_emb_dim"], xcfg["d_model"], xcfg["heads"],
                         xcfg["layers"], xcfg["ff"], xcfg["dropout"], xcfg.get("zero_mean", False))

    preds, best_epochs = {}, []
    for seed in seeds_of(xcfg):
        if verbose:
            print(f" seed {seed}")
        preds[seed], ep = _fit_seed(gtr, gva, yva, xcfg, n_stocks, seed, epochs, verbose, build)
        best_epochs.append(ep)
    pred = np.mean(list(preds.values()), axis=0)
    return pred, None if fixed else mae(yva, pred), best_epochs, preds


def main():
    cli("xformer", train_fold)


if __name__ == "__main__":
    main()
