"""README 頂端的結果圖：每一步相對「預測 0」改善了多少（CV 與 holdout 並列）。

    python scripts/plot_results.py --out reports/results.png
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

# (標籤, 結果資料夾, 取哪個分數)；每一步都在前一步之上累加
STAGES = [
    ("LightGBM", "reports", "lgb"),
    ("+ MLP ensemble", "reports", "ensemble"),
    ("+ 指數中性後處理", "reports", "neutral"),
    ("+ 擴充時序特徵", "reports/temporal_wide", "neutral"),
]
CV_COLOR, HO_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def scores(d: Path, kind: str) -> tuple[float, float | None]:
    """回傳 (CV 改善 %, holdout 改善 %)；沒有 holdout 時後者為 None。"""
    r = json.loads((d / "ensemble_report.json").read_text())
    zero_cv = json.loads((d / "summary_lgb.json").read_text())["cv_mae_zero"]
    cv = {"lgb": r["single"]["lgb"], "ensemble": r["weighted_nested"],
          "neutral": r["neutral"]["weighted_nested"]}[kind]
    ho = None
    if "holdout" in r:
        h = r["holdout"]
        ho = {"lgb": h["single"]["lgb"], "ensemble": h["weighted"],
              "neutral": h["neutral"]["weighted"]}[kind]
        ho = (h["predict_zero"] - ho) / h["predict_zero"] * 100
    return (zero_cv - cv) / zero_cv * 100, ho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/results.png")
    args = ap.parse_args()
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 找不到的備用字型不用警告
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "DejaVu Sans"]

    rows = [(label, *scores(Path(d), kind)) for label, d, kind in STAGES]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    # 橫軸不從 0 開始，用點而不是長條，避免長度誇大差距
    for i, (label, cv, ho) in enumerate(rows):
        y = len(rows) - 1 - i
        if ho is not None:
            ax.plot([min(cv, ho), max(cv, ho)], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter(ho, y, s=70, color=HO_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3,
                       label="Holdout" if i == 0 else None)
            ax.annotate(f"{ho:.2f}%", (ho, y), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8.5, color=MUTED)
        ax.scatter(cv, y, s=70, color=CV_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="CV" if i == 0 else None)
        ax.annotate(f"{cv:.2f}%", (cv, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
    lo = min(v for r in rows for v in r[1:] if v is not None)
    hi = max(v for r in rows for v in r[1:] if v is not None)
    ax.set_xlim(lo - 0.05, hi + 0.05)  # 差距只有 0.3 個百分點，從 0 畫會看不出來
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("MAE 相對「預測 0」的改善（%，越大越好）", color=MUTED, fontsize=9)
    ax.set_title("每一步的改善", loc="left", color=INK, fontsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    hs, ls = ax.get_legend_handles_labels()
    ax.legend(hs[::-1], ls[::-1], loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
