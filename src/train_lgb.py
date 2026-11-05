"""LightGBM baseline：walk-forward CV、OOF 預測、特徵重要性，以及只用一次的 holdout 評估。

    python -m src.train_lgb --config configs/base.yaml
    python -m src.train_lgb --config configs/base.yaml --drop-groups temporal,index --tag no_temporal
    python -m src.train_lgb --config configs/base.yaml --holdout     # 第 7 週才跑，且只跑一次
    python -m src.train_lgb --config configs/lgb_refit.yaml   # 設定 refit 時改跑定期重訓的 CV
    python -m src.train_lgb --config configs/base.yaml --holdout-rounds 557  #跳過 CV，直接用指定輪數
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import load_config
from .features import feature_groups
from .pipeline import Prepared, fold_frames, mae, prepare


def _split_params(lcfg: dict):
    params = {k: v for k, v in lcfg.items() if k not in ("num_boost_round", "early_stopping_rounds")}
    return params, lcfg["num_boost_round"], lcfg["early_stopping_rounds"]


def _select(feats: list[str], drop_groups: list[str]) -> list[str]:
    if not drop_groups:
        return feats
    groups = feature_groups(feats)
    drop = {c for g in drop_groups for c in groups.get(g, [])}
    return [c for c in feats if c not in drop]


def run_cv(p: Prepared, cfg: dict, drop_groups: list[str] | None = None, verbose: bool = True) -> dict:
    drop_groups = drop_groups or []
    params, n_rounds, es = _split_params(cfg["lgb"])
    use_ss = "stock" in cfg["features"]["groups"] and "stock" not in drop_groups
    oof, rows, importances = [], [], []
    for f in p.folds:
        t0 = time.time()
        tr, va, feats = fold_frames(p, f.train_dates, f.val_dates, use_ss)
        feats = _select(feats, drop_groups)
        dtr = lgb.Dataset(tr[feats], tr["target"], free_raw_data=True)
        dva = lgb.Dataset(va[feats], va["target"], reference=dtr, free_raw_data=True)
        model = lgb.train(params, dtr, n_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(es, verbose=False)])
        pred = model.predict(va[feats], num_iteration=model.best_iteration)
        score = mae(va["target"], pred)
        rows.append({"fold": f.fold, "train_days": len(f.train_dates), "val_days": len(f.val_dates),
                     "val_start": int(f.val_dates[0]), "val_end": int(f.val_dates[-1]),
                     "best_iter": model.best_iteration, "mae": score,
                     "mae_zero": mae(va["target"], 0.0), "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"fold {f.fold}: MAE {score:.5f} (predict-0 {rows[-1]['mae_zero']:.5f}) "
                  f"iter {model.best_iteration} [{rows[-1]['seconds']}s]")
        oof.append(va[["stock_id", "date_id", "seconds_in_bucket", "target"]]
                   .assign(pred=pred.astype(np.float32), fold=f.fold))
        importances.append(pd.Series(model.feature_importance("gain"), index=feats, name=f.fold))
    folds = pd.DataFrame(rows)
    oof = pd.concat(oof, ignore_index=True)
    return {
        "folds": folds,
        "oof": oof,
        "importance": pd.concat(importances, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False),
        "cv_mae": mae(oof["target"], oof["pred"]),
        "cv_mae_zero": mae(oof["target"], 0.0),
        "mean_best_iter": int(folds["best_iter"].mean()),
        "n_features": len(feats),
    }


def refit_schedule(train_dates: np.ndarray, val_dates: np.ndarray, every_days: int,
                   gap_days: int = 0, embargo_dates: np.ndarray | None = None
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    """驗證期每 every_days 天切一段；每段的訓練資料 = 原訓練日期 + 這段開始前已揭露的日期。

    比賽 API 每天開盤前公布前一天的 target，所以第 d 天可以用到 d-1 天（含）以前的資料；
    gap_days > 0 時再往前多空幾天。第一段就是原本的訓練日期，與一般 CV 相同；
    之後各段也加入 embargo 那幾天（訓練集與驗證集之間），上線時它們早已揭露。"""
    emb = np.asarray([] if embargo_dates is None else embargo_dates, dtype=val_dates.dtype)
    out = []
    for s in range(0, len(val_dates), every_days):
        start = val_dates[s]
        known = np.concatenate([emb, val_dates[:s]])
        seen = known[known < start - gap_days] if s else known[:0]
        out.append((np.concatenate([train_dates, seen]), val_dates[s:s + every_days]))
    return out


def run_refit_cv(p: Prepared, cfg: dict, verbose: bool = True) -> dict:
    """定期重訓的 CV：每折先照一般 CV 訓練（驗證集 early stopping，得到輪數 B），
    第一段直接用這個模型；之後每段用已揭露的驗證日期加進訓練集、固定 B 輪重訓，只預測這一段。
    同一個初始模型對整個驗證期的預測也存下來（pred_static），兩者只差在有沒有重訓。"""
    params, n_rounds, es = _split_params(cfg["lgb"])
    rcfg = cfg["refit"]
    use_ss = "stock" in cfg["features"]["groups"]
    cv_dates = np.setdiff1d(np.unique(p.df["date_id"]), p.holdout_dates)
    oof, rows = [], []
    for f in p.folds:
        t0 = time.time()
        tr, va, feats = fold_frames(p, f.train_dates, f.val_dates, use_ss)
        dtr = lgb.Dataset(tr[feats], tr["target"], free_raw_data=True)
        dva = lgb.Dataset(va[feats], va["target"], reference=dtr, free_raw_data=True)
        model = lgb.train(params, dtr, n_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(es, verbose=False)])
        best = model.best_iteration
        keep = va[["stock_id", "date_id", "seconds_in_bucket", "target"]].assign(
            pred_static=model.predict(va[feats], num_iteration=best).astype(np.float32), fold=f.fold)
        del tr, dtr, dva
        keep["pred"] = np.nan
        emb = cv_dates[(cv_dates > f.train_dates[-1]) & (cv_dates < f.val_dates[0])]
        sched = refit_schedule(f.train_dates, f.val_dates, rcfg["every_days"], rcfg.get("gap_days", 0), emb)
        for k, (fit_dates, pred_dates) in enumerate(sched):
            m = keep["date_id"].isin(pred_dates).to_numpy()
            if k == 0:
                keep.loc[m, "pred"] = keep.loc[m, "pred_static"]
                continue
            t1 = time.time()
            # 每段重算 stock 統計量，只用這段之前的資料
            tr_k, va_k, feats_k = fold_frames(p, fit_dates, pred_dates, use_ss)
            mk = lgb.train(params, lgb.Dataset(tr_k[feats_k], tr_k["target"]), best)
            # 兩邊都是從 p.df 依原順序篩出來的（stock 統計量的 merge 會重設 index），逐列對齊
            key = ["stock_id", "date_id", "seconds_in_bucket"]
            assert (va_k[key].to_numpy() == keep.loc[m, key].to_numpy()).all()
            keep.loc[m, "pred"] = mk.predict(va_k[feats_k]).astype(np.float32)
            if verbose:
                print(f"  refit {k}: +{len(fit_dates) - len(f.train_dates)} days, "
                      f"predict {pred_dates[0]}-{pred_dates[-1]} [{time.time() - t1:.0f}s]")
            del tr_k, va_k
        y = keep["target"]
        rows.append({"fold": f.fold, "best_iter": best, "n_refits": len(sched) - 1,
                     "mae_static": mae(y, keep["pred_static"]), "mae": mae(y, keep["pred"]),
                     "mae_zero": mae(y, 0.0), "seconds": round(time.time() - t0, 1)})
        if verbose:
            r = rows[-1]
            print(f"fold {f.fold}: MAE refit {r['mae']:.5f} vs static {r['mae_static']:.5f} "
                  f"({r['mae'] - r['mae_static']:+.5f}) iter {best} [{r['seconds']}s]")
        oof.append(keep)
    folds = pd.DataFrame(rows)
    oof = pd.concat(oof, ignore_index=True)
    return {"folds": folds, "oof": oof,
            "cv_mae": mae(oof["target"], oof["pred"]),
            "cv_mae_static": mae(oof["target"], oof["pred_static"]),
            "cv_mae_zero": mae(oof["target"], 0.0),
            "mean_best_iter": int(folds["best_iter"].mean()),
            "refit": rcfg}


def run_holdout(p: Prepared, cfg: dict, n_rounds: int, drop_groups, out: Path) -> float:
    log = out / "holdout_log.csv"
    if log.exists() and (pd.read_csv(log)["model"] == "lgb").any():
        print(f"警告：lgb 的 holdout 已經用過（見 {log}）。重複使用會讓它失去意義。")
    cv_dates = np.sort(np.concatenate([p.folds[-1].train_dates, p.folds[-1].val_dates]))
    emb = cfg["cv"]["embargo_days"]
    train_dates = cv_dates[:-emb] if emb else cv_dates
    use_ss = "stock" in cfg["features"]["groups"] and "stock" not in (drop_groups or [])
    tr, ho, feats = fold_frames(p, train_dates, p.holdout_dates, use_ss)
    feats = _select(feats, drop_groups)
    params, _, _ = _split_params(cfg["lgb"])
    model = lgb.train(params, lgb.Dataset(tr[feats], tr["target"]), n_rounds)
    pred = model.predict(ho[feats])
    # 存下預測，供 ensemble 計算 holdout 分數
    ho[["stock_id", "date_id", "seconds_in_bucket", "target"]].assign(
        pred=pred.astype(np.float32)).to_parquet(out / "holdout_pred_lgb.parquet")
    score = mae(ho["target"], pred)
    entry = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "model": "lgb",
                           "n_rounds": n_rounds, "holdout_mae": score,
                           "holdout_mae_zero": mae(ho["target"], 0.0)}])
    entry.to_csv(log, mode="a", header=not log.exists(), index=False)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--drop-groups", default="", help="以逗號分隔，例如 temporal,index")
    ap.add_argument("--tag", default="lgb")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--holdout-rounds", type=int, default=None,
                    help="跳過 CV，直接以此輪數訓練並評估 holdout")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    drop = [g for g in args.drop_groups.split(",") if g]

    p = prepare(cfg)
    print(f"{len(p.df):,} rows, {len(p.features)} features, {len(p.folds)} folds, "
          f"holdout {len(p.holdout_dates)} days")
    if args.holdout_rounds:
        score = run_holdout(p, cfg, args.holdout_rounds, drop, out)
        print(f"Holdout MAE {score:.5f}")
        return
    if cfg.get("refit"):
        res = run_refit_cv(p, cfg)
        print(f"CV MAE refit {res['cv_mae']:.5f} | static {res['cv_mae_static']:.5f} "
              f"| predict-0 {res['cv_mae_zero']:.5f}")
        res["folds"].to_csv(out / f"cv_{args.tag}.csv", index=False)
        res["oof"].to_parquet(out / f"oof_{args.tag}.parquet")
        (out / f"summary_{args.tag}.json").write_text(json.dumps(
            {k: res[k] for k in ("cv_mae", "cv_mae_static", "cv_mae_zero", "mean_best_iter", "refit")},
            indent=2))
        return
    res = run_cv(p, cfg, drop)
    print(f"CV MAE {res['cv_mae']:.5f} | predict-0 {res['cv_mae_zero']:.5f}")

    res["folds"].to_csv(out / f"cv_{args.tag}.csv", index=False)
    res["oof"].to_parquet(out / f"oof_{args.tag}.parquet")
    res["importance"].rename("gain").to_csv(out / f"importance_{args.tag}.csv")
    summary = {k: res[k] for k in ("cv_mae", "cv_mae_zero", "mean_best_iter", "n_features")}
    summary["drop_groups"] = drop
    if args.holdout:
        summary["holdout_mae"] = run_holdout(p, cfg, res["mean_best_iter"], drop, out)
        print(f"Holdout MAE {summary['holdout_mae']:.5f}")
    (out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
