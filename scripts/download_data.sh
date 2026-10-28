#!/usr/bin/env bash
# 下載比賽資料到 data/。需要先：
#   1. 在 Kaggle 比賽頁按 "Late Submission" / "Join Competition" 同意規則
#   2. 在 Kaggle 帳號設定頁產生 API token，放到 ~/.kaggle/kaggle.json（chmod 600）
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
kaggle competitions download -c optiver-trading-at-the-close -p data
unzip -o data/optiver-trading-at-the-close.zip -d data
ls -lh data
