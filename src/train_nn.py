"""MLP：同一套特徵 + stock_id embedding，和 LightGBM 用完全相同的折。

    python -m src.train_nn --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn

from .config import load_config
from .pipeline import fold_frames, mae, prepare


class MLP(nn.Module):
    def __init__(self, n_num: int, n_stocks: int, emb_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_stocks, emb_dim)
        layers, d = [], n_num + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_stock):
        return self.net(torch.cat([x_num, self.emb(x_stock)], dim=1)).squeeze(1)


class Preprocessor:
    """統計量只從訓練折 fit：rank-gauss（或標準化）→ clip → 缺值補 0。"""

    def __init__(self, kind: str, seed: int):
        self.kind, self.seed = kind, seed

    def fit(self, X: np.ndarray):
        if self.kind == "quantile":
            self.t = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                         subsample=200_000, random_state=self.seed)
        else:
            self.t = StandardScaler()
        self.t.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.t.transform(X)
        return np.nan_to_num(np.clip(Z, -5, 5), nan=0.0).astype(np.float32)


def _numeric(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = df[feats].to_numpy(np.float64)
    X[~np.isfinite(X)] = np.nan
    return X


def train_fold(tr, va, feats, ncfg, n_stocks, verbose=True):
    torch.manual_seed(ncfg["seed"])
    num = [c for c in feats if c != "stock_id"]
    pre = Preprocessor(ncfg["scaler"], ncfg["seed"]).fit(_numeric(tr, num))
    Xtr = torch.from_numpy(pre.transform(_numeric(tr, num)))
    Xva = torch.from_numpy(pre.transform(_numeric(va, num)))
    Str = torch.from_numpy(tr["stock_id"].to_numpy(np.int64))
    Sva = torch.from_numpy(va["stock_id"].to_numpy(np.int64))
    ytr = torch.from_numpy(tr["target"].to_numpy(np.float32))
    yva = va["target"].to_numpy(np.float32)

    model = MLP(len(num), n_stocks, ncfg["stock_emb_dim"], ncfg["hidden"], ncfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.L1Loss()  # 直接對齊 MAE 評分
    bs = ncfg["batch_size"]

    def predict(X, S):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(X[i:i + 65536], S[i:i + 65536])
                              for i in range(0, len(X), 65536)]).numpy()

    best, best_state, bad = np.inf, None, 0
    for epoch in range(ncfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Str[idx]), ytr[idx])
            loss.backward()
            opt.step()
        score = mae(yva, predict(Xva, Sva))
        sched.step(score)
        if verbose:
            print(f"  epoch {epoch}: val MAE {score:.5f}")
        if score < best - 1e-5:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ncfg["patience"]:
                break
    model.load_state_dict(best_state)
    return predict(Xva, Sva), best, epoch + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tag", default="nn")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["nn"].get("num_threads"):
        torch.set_num_threads(cfg["nn"]["num_threads"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    p = prepare(cfg)
    n_stocks = int(p.df["stock_id"].max()) + 1
    use_ss = "stock" in cfg["features"]["groups"]
    oof, rows = [], []
    for f in p.folds:
        t0 = time.time()
        tr, va, feats = fold_frames(p, f.train_dates, f.val_dates, use_ss)
        print(f"fold {f.fold}")
        pred, score, epochs = train_fold(tr, va, feats, cfg["nn"], n_stocks)
        rows.append({"fold": f.fold, "mae": score, "epochs": epochs,
                     "seconds": round(time.time() - t0, 1)})
        print(f"fold {f.fold}: MAE {score:.5f} [{rows[-1]['seconds']}s]")
        oof.append(va[["stock_id", "date_id", "seconds_in_bucket", "target"]]
                   .assign(pred=pred.astype(np.float32), fold=f.fold))
    oof = pd.concat(oof, ignore_index=True)
    cv = mae(oof["target"], oof["pred"])
    print(f"CV MAE {cv:.5f}")
    pd.DataFrame(rows).to_csv(out / f"cv_{args.tag}.csv", index=False)
    oof.to_parquet(out / f"oof_{args.tag}.parquet")
    (out / f"summary_{args.tag}.json").write_text(json.dumps({"cv_mae": cv}, indent=2))


if __name__ == "__main__":
    main()
