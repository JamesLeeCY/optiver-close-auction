"""GRU 序列模型：同一檔股票當天的 55 個時點當成一條序列，單向 GRU 逐格預測 target。

單向 GRU 在第 t 格的輸出只看得到第 0..t 格，與表格特徵「只往前看」的規則相同。
預設拿掉 temporal 特徵組（lag、diff、rolling mean 等），時間上的變化交給 GRU 自己學，
讓它和 LightGBM / MLP 的差異更大。和其他模型用完全相同的折。

    python -m src.train_gru --config configs/gru.yaml
    python -m src.train_gru --config configs/gru.yaml --holdout-epochs 8   # 跳過 CV，直接用指定 epoch 數
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
from torch import nn

from .config import load_config
from .features import feature_groups
from .pipeline import fold_frames, mae, prepare
from .train_nn import Preprocessor, _numeric, seeds_of

STEP = 10  # seconds_in_bucket 每 10 秒一格


class GRUNet(nn.Module):
    def __init__(self, n_num: int, n_stocks: int, emb_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_stocks, emb_dim)
        self.inp = nn.Sequential(nn.Linear(n_num + emb_dim, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x, s):
        """x: [B, T, F]，s: [B] → [B, T]"""
        e = self.emb(s)[:, None, :].expand(-1, x.shape[1], -1)
        h = self.inp(torch.cat([x, e], dim=2))
        o, _ = self.gru(h)
        return self.head(torch.cat([o, h], dim=2)).squeeze(-1)


class Sequences:
    """把列（stock, date, 秒）排成 [序列, 時點, 特徵]；缺的時點補 0、不算進 loss。"""

    def __init__(self, df: pd.DataFrame, X: np.ndarray, n_steps: int):
        key = df["date_id"].to_numpy(np.int64) * 100_000 + df["stock_id"].to_numpy(np.int64)
        uniq, self.seq = np.unique(key, return_inverse=True)
        self.step = (df["seconds_in_bucket"].to_numpy() // STEP).astype(np.int64)
        self.x = np.zeros((len(uniq), n_steps, X.shape[1]), dtype=np.float32)
        self.x[self.seq, self.step] = X
        self.y = np.zeros((len(uniq), n_steps), dtype=np.float32)
        self.y[self.seq, self.step] = df["target"].to_numpy(np.float32)
        self.mask = np.zeros((len(uniq), n_steps), dtype=bool)
        self.mask[self.seq, self.step] = True
        self.stock = (uniq % 100_000).astype(np.int64)

    def to_rows(self, p: np.ndarray) -> np.ndarray:
        """[序列, 時點] 的預測 → 對回原本列的順序。"""
        return p[self.seq, self.step]


def gru_features(feats: list[str], gcfg: dict) -> list[str]:
    drop = {c for g in gcfg.get("drop_groups", []) for c in feature_groups(feats).get(g, [])}
    return [c for c in feats if c not in drop]


def train_fold(tr, va, feats, gcfg, n_stocks, n_steps, epochs=None, verbose=True):
    """與 train_nn.train_fold 相同的介面：前處理只做一次，每個種子各訓練一個模型、預測取平均。
    epochs 為 None 時在 va 上 early stopping；給定 epochs 時固定輪數、不看 va 的 target（用於 holdout）。"""
    fixed = epochs is not None
    num = [c for c in gru_features(feats, gcfg) if c != "stock_id"]
    Xtr_raw = _numeric(tr, num)
    pre = Preprocessor(gcfg["scaler"], gcfg["seed"]).fit(Xtr_raw)
    str_ = Sequences(tr, pre.transform(Xtr_raw), n_steps)
    del Xtr_raw
    sva = Sequences(va, pre.transform(_numeric(va, num)), n_steps)
    yva = None if fixed else va["target"].to_numpy(np.float32)
    if verbose:
        print(f" {len(num)} features, {len(str_.x)} train / {len(sva.x)} val sequences")

    preds, best_epochs = {}, []
    for seed in seeds_of(gcfg):
        if verbose:
            print(f" seed {seed}")
        preds[seed], ep = _fit_seed(str_, sva, yva, gcfg, n_stocks, seed, epochs, verbose)
        best_epochs.append(ep)
    pred = np.mean(list(preds.values()), axis=0)
    return pred, None if fixed else mae(yva, pred), best_epochs, preds


def _fit_seed(str_: Sequences, sva: Sequences, yva, gcfg, n_stocks, seed, epochs=None, verbose=True):
    torch.manual_seed(seed)
    fixed = epochs is not None
    Xtr, Ytr = torch.from_numpy(str_.x), torch.from_numpy(str_.y)
    Mtr, Str = torch.from_numpy(str_.mask), torch.from_numpy(str_.stock)
    Xva, Sva = torch.from_numpy(sva.x), torch.from_numpy(sva.stock)
    model = GRUNet(Xtr.shape[2], n_stocks, gcfg["stock_emb_dim"], gcfg["hidden"],
                   gcfg["layers"], gcfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    bs = gcfg["batch_size"]

    def predict():
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(Xva[i:i + 2048], Sva[i:i + 2048]) for i in range(0, len(Xva), 2048)])
        return sva.to_rows(p.numpy())

    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(epochs if fixed else gcfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr))
        total, n = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            m = Mtr[idx]
            opt.zero_grad()
            err = (model(Xtr[idx], Str[idx]) - Ytr[idx]).abs()
            loss = err[m].mean()  # 只算有資料的時點，直接對齊 MAE
            loss.backward()
            opt.step()
            total += loss.item() * int(m.sum())
            n += int(m.sum())
        if fixed:
            sched.step(total / n)
            if verbose:
                print(f"  epoch {epoch}: train MAE {total / n:.5f}")
            continue
        score = mae(yva, predict())
        sched.step(score)
        if verbose:
            print(f"  epoch {epoch}: train MAE {total / n:.5f} val MAE {score:.5f}")
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= gcfg["patience"]:
                break
    if fixed:
        return predict(), epochs
    model.load_state_dict(best_state)
    if verbose:
        print(f"  best val MAE {best:.5f} @ epoch {best_epoch}")
    return predict(), best_epoch


def run_holdout(p, cfg, n_stocks, n_steps, epochs: int, out: Path) -> float:
    log = out / "holdout_log.csv"
    if log.exists() and (pd.read_csv(log)["model"] == "gru").any():
        print(f"警告：gru 的 holdout 已經用過（見 {log}）。重複使用會讓它失去意義。")
    # 與 train_nn.run_holdout 相同：全部 CV 日期去掉最後 embargo 天
    cv_dates = np.sort(np.concatenate([p.folds[-1].train_dates, p.folds[-1].val_dates]))
    emb = cfg["cv"]["embargo_days"]
    train_dates = cv_dates[:-emb] if emb else cv_dates
    tr, ho, feats = fold_frames(p, train_dates, p.holdout_dates, "stock" in cfg["features"]["groups"])
    print(f"holdout: train {len(train_dates)} days, {epochs} epochs, holdout {len(p.holdout_dates)} days")
    pred, _, _, _ = train_fold(tr, ho, feats, cfg["gru"], n_stocks, n_steps, epochs=epochs)
    ho[["stock_id", "date_id", "seconds_in_bucket", "target"]].assign(
        pred=pred.astype(np.float32)).to_parquet(out / "holdout_pred_gru.parquet")
    score = mae(ho["target"], pred)
    entry = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "model": "gru",
                           "n_rounds": epochs, "holdout_mae": score,
                           "holdout_mae_zero": mae(ho["target"], 0.0)}])
    entry.to_csv(log, mode="a", header=not log.exists(), index=False)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gru.yaml")
    ap.add_argument("--tag", default="gru")
    ap.add_argument("--holdout-epochs", type=int, default=None,
                    help="跳過 CV，直接以此 epoch 數訓練並評估 holdout")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gcfg = cfg["gru"]
    if gcfg.get("num_threads"):
        torch.set_num_threads(gcfg["num_threads"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    p = prepare(cfg)
    n_stocks = int(p.df["stock_id"].max()) + 1
    n_steps = int(p.df["seconds_in_bucket"].max()) // STEP + 1
    if args.holdout_epochs:
        score = run_holdout(p, cfg, n_stocks, n_steps, args.holdout_epochs, out)
        print(f"Holdout MAE {score:.5f}")
        return
    use_ss = "stock" in cfg["features"]["groups"]
    oof, rows = [], []
    for f in p.folds:
        t0 = time.time()
        tr, va, feats = fold_frames(p, f.train_dates, f.val_dates, use_ss)
        print(f"fold {f.fold}")
        pred, score, epochs, by_seed = train_fold(tr, va, feats, gcfg, n_stocks, n_steps)
        y = va["target"].to_numpy()
        rows.append({"fold": f.fold, "mae": score, "best_epoch": float(np.mean(epochs)),
                     **{f"mae_s{s}": mae(y, q) for s, q in by_seed.items()},
                     "seconds": round(time.time() - t0, 1)})
        print(f"fold {f.fold}: MAE {score:.5f} [{rows[-1]['seconds']}s]")
        oof.append(va[["stock_id", "date_id", "seconds_in_bucket", "target"]]
                   .assign(pred=pred.astype(np.float32), fold=f.fold))
    oof = pd.concat(oof, ignore_index=True)
    cv = mae(oof["target"], oof["pred"])
    print(f"CV MAE {cv:.5f}")
    pd.DataFrame(rows).to_csv(out / f"cv_{args.tag}.csv", index=False)
    oof.to_parquet(out / f"oof_{args.tag}.parquet")
    summary = {"cv_mae": cv, "mean_best_epoch": int(round(np.mean([r["best_epoch"] for r in rows]))),
               "seeds": seeds_of(gcfg), "n_features": len(gru_features(p.features, gcfg))}
    (out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
