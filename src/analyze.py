"""特徵重要性分析：ablation（拿掉一組特徵看 CV 掉多少）與 SHAP。

    python -m src.analyze ablation --config configs/base.yaml
    python -m src.analyze shap --config configs/base.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import load_config
from .features import feature_groups
from .pipeline import fold_frames, prepare
from .train_lgb import _split_params, run_cv

GROUPS = ["basic", "auction", "time", "temporal", "cross_sectional", "index", "stock"]


def ablation(cfg: dict, out: Path):
    p = prepare(cfg)
    base = run_cv(p, cfg, verbose=False)
    rows = [{"dropped": "(none)", "cv_mae": base["cv_mae"], "delta": 0.0,
             "n_features": base["n_features"]}]
    print(f"full: {base['cv_mae']:.5f}")
    for g in GROUPS:
        if g not in cfg["features"]["groups"]:
            continue
        r = run_cv(p, cfg, [g], verbose=False)
        rows.append({"dropped": g, "cv_mae": r["cv_mae"], "delta": r["cv_mae"] - base["cv_mae"],
                     "n_features": r["n_features"]})
        print(f"-{g}: {r['cv_mae']:.5f} (Δ {rows[-1]['delta']:+.5f})")
        # 每組跑完就存：全量一次要好幾小時，中途中斷也保留已完成的部分
        pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation.csv", index=False)
    (out / "ablation.md").write_text(table.to_markdown(index=False, floatfmt=".5f"))


def shap_report(cfg: dict, out: Path, n_sample: int = 20_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    p = prepare(cfg)
    f = p.folds[-1]
    tr, va, feats = fold_frames(p, f.train_dates, f.val_dates, "stock" in cfg["features"]["groups"])
    params, n_rounds, es = _split_params(cfg["lgb"])
    model = lgb.train(params, lgb.Dataset(tr[feats], tr["target"]), n_rounds,
                      valid_sets=[lgb.Dataset(va[feats], va["target"])],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
    X = va[feats].sample(min(n_sample, len(va)), random_state=0)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(out / "shap_importance.csv")

    groups = feature_groups(feats)
    by_group = {g: imp.reindex(cols).sum() for g, cols in groups.items() if cols}
    pd.Series(by_group).sort_values(ascending=False).rename("mean_abs_shap") \
        .to_csv(out / "shap_by_group.csv")

    shap.summary_plot(sv, X, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=150)
    print(imp.head(20).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ablation", "shap"])
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ablation(cfg, out) if args.what == "ablation" else shap_report(cfg, out)


if __name__ == "__main__":
    main()
