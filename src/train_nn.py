"""MLP：同一套特徵 + stock_id embedding，和 LightGBM 用完全相同的折。

    python -m src.train_nn --config configs/base.yaml
    python -m src.train_nn --config configs/base.yaml --holdout            # CV 後再評估 holdout（只跑一次）
    python -m src.train_nn --config configs/base.yaml --holdout-epochs 10  # 跳過 CV，直接用指定 epoch 數
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
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
        # 以 float64 fit：float32 算出的分位數會讓剛好落在分位點上的值對應到不同結果
        self.t.fit(X.astype(np.float64))
        return self

    def transform(self, X: np.ndarray, chunk: int = 500_000) -> np.ndarray:
        # 分批轉換、直接寫進 float32：全量一折 400 萬列，一次轉會產生好幾份 float64 暫存
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, len(X), chunk):
            Z = self.t.transform(X[i:i + chunk].astype(np.float64))
            out[i:i + chunk] = np.nan_to_num(np.clip(Z, -5, 5), nan=0.0)
        return out


def _numeric(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = df[feats].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def train_fold(tr, va, feats, ncfg, n_stocks, epochs=None, verbose=True):
    """epochs 為 None：在 va 上 early stopping。給定 epochs：固定訓練這麼多輪、不看 va 的 target
    （用於 holdout），學習率改依訓練 loss 調整。"""
    torch.manual_seed(ncfg["seed"])
    fixed = epochs is not None
    num = [c for c in feats if c != "stock_id"]
    Xtr_raw = _numeric(tr, num)
    pre = Preprocessor(ncfg["scaler"], ncfg["seed"]).fit(Xtr_raw)
    Xtr = torch.from_numpy(pre.transform(Xtr_raw))
    del Xtr_raw
    Xva = torch.from_numpy(pre.transform(_numeric(va, num)))
    Str = torch.from_numpy(tr["stock_id"].to_numpy(np.int64))
    Sva = torch.from_numpy(va["stock_id"].to_numpy(np.int64))
    ytr = torch.from_numpy(tr["target"].to_numpy(np.float32))
    yva = None if fixed else va["target"].to_numpy(np.float32)

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

    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(epochs if fixed else ncfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Str[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if fixed:
            train_loss = total / len(perm)
            sched.step(train_loss)
            if verbose:
                print(f"  epoch {epoch}: train MAE {train_loss:.5f}")
            continue
        score = mae(yva, predict(Xva, Sva))
        sched.step(score)
        if verbose:
            print(f"  epoch {epoch}: val MAE {score:.5f}")
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ncfg["patience"]:
                break
    if fixed:
        return predict(Xva, Sva), None, epochs
    model.load_state_dict(best_state)
    return predict(Xva, Sva), best, best_epoch


def run_holdout(p, cfg, n_stocks, epochs: int, out: Path) -> float:
    log = out / "holdout_log.csv"
    if log.exists() and (pd.read_csv(log)["model"] == "nn").any():
        print(f"警告：nn 的 holdout 已經用過（見 {log}）。重複使用會讓它失去意義。")
    # 與 train_lgb.run_holdout 相同：全部 CV 日期去掉最後 embargo 天
    cv_dates = np.sort(np.concatenate([p.folds[-1].train_dates, p.folds[-1].val_dates]))
    emb = cfg["cv"]["embargo_days"]
    train_dates = cv_dates[:-emb] if emb else cv_dates
    tr, ho, feats = fold_frames(p, train_dates, p.holdout_dates, "stock" in cfg["features"]["groups"])
    print(f"holdout: train {len(train_dates)} days, {epochs} epochs, holdout {len(p.holdout_dates)} days")
    pred, _, _ = train_fold(tr, ho, feats, cfg["nn"], n_stocks, epochs=epochs)
    # 存下預測，供 ensemble 計算 holdout 分數
    ho[["stock_id", "date_id", "seconds_in_bucket", "target"]].assign(
        pred=pred.astype(np.float32)).to_parquet(out / "holdout_pred_nn.parquet")
    score = mae(ho["target"], pred)
    entry = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "model": "nn",
                           "n_rounds": epochs, "holdout_mae": score,
                           "holdout_mae_zero": mae(ho["target"], 0.0)}])
    entry.to_csv(log, mode="a", header=not log.exists(), index=False)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--holdout", action="store_true", help="CV 後以平均最佳 epoch 數評估 holdout")
    ap.add_argument("--holdout-epochs", type=int, default=None,
                    help="跳過 CV，直接以此 epoch 數訓練並評估 holdout")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["nn"].get("num_threads"):
        torch.set_num_threads(cfg["nn"]["num_threads"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    p = prepare(cfg)
    n_stocks = int(p.df["stock_id"].max()) + 1
    if args.holdout_epochs:
        score = run_holdout(p, cfg, n_stocks, args.holdout_epochs, out)
        print(f"Holdout MAE {score:.5f}")
        return
    use_ss = "stock" in cfg["features"]["groups"]
    oof, rows = [], []
    for f in p.folds:
        t0 = time.time()
        tr, va, feats = fold_frames(p, f.train_dates, f.val_dates, use_ss)
        print(f"fold {f.fold}")
        pred, score, epochs = train_fold(tr, va, feats, cfg["nn"], n_stocks)
        rows.append({"fold": f.fold, "mae": score, "best_epoch": epochs,
                     "seconds": round(time.time() - t0, 1)})
        print(f"fold {f.fold}: MAE {score:.5f} [{rows[-1]['seconds']}s]")
        oof.append(va[["stock_id", "date_id", "seconds_in_bucket", "target"]]
                   .assign(pred=pred.astype(np.float32), fold=f.fold))
    oof = pd.concat(oof, ignore_index=True)
    cv = mae(oof["target"], oof["pred"])
    print(f"CV MAE {cv:.5f}")
    pd.DataFrame(rows).to_csv(out / f"cv_{args.tag}.csv", index=False)
    oof.to_parquet(out / f"oof_{args.tag}.parquet")
    summary = {"cv_mae": cv, "mean_best_epoch": int(round(np.mean([r["best_epoch"] for r in rows])))}
    if args.holdout:
        summary["holdout_mae"] = run_holdout(p, cfg, n_stocks, summary["mean_best_epoch"], out)
        print(f"Holdout MAE {summary['holdout_mae']:.5f}")
    (out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
