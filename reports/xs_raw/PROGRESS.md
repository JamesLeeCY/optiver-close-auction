# 橫斷面排名與時段分組特徵：進度紀錄（2026-11-06 23:20）

## 已完成
- `src/features.py` 新增兩組特徵（參考 Kaggle 第一名 hyd），共 41 欄，特徵數 235 → 276：
  - `xs_raw`：10 個原始欄位（imbalance_size、reference_price、matched_size、far/near_price、
    bid/ask_price、bid/ask_size、wap）在同一時點的排名（`_xs_rank`）與相對平均的倍數（`_xs_mratio`）
  - `segment`：一天切成 0–300 / 300–480 / 480 秒後三段（`seg_id`），同樣 10 欄相對「該段至今第一個值」
    （`_seg_first_ratio`）與「該段至今平均」（`_seg_exp_ratio`）的倍數，只用含當下以前的資料
- 洩漏測試（`tests/test_leakage.py`）：在 250 / 400 / 500 秒切開竄改未來、只改其他時點、段首缺值，27 個測試全過
- 設定：`configs/xs_raw.yaml`（LightGBM）、`xformer_xs_raw.yaml`（2 層、零均值、3 種子）、
  `xformer_xs_raw_l5.yaml`（5 層）
- 全量特徵快取已建好：`data/cache/features_b8c81ef87e.parquet`（3.6 GB），重跑不用再建

## 進行中 / 待辦
- **LightGBM CV（本機）**：23:14 開始訓練 fold 0 時，Ollama（llama-server）佔約 7 個核心，CPU 滿載，
  手動停止（沒有任何一折完成）。改由等待程式在 CPU 空出來後自動重跑：從 11-07 01:00 起開始檢查（另一個專案預計 03:00 左右跑完），連續 5 分鐘整體 CPU ≤ 40% 且
  可用 RAM ≥ 20 GB 才開始，最多等 12 小時，紀錄在 `reports/xs_raw/waiter.log`。
  比較基準：temporal_wide LightGBM（學習率 0.05、不重訓）CV 6.15251，每折 6.24828 / 5.90264 / 6.38960 /
  6.25755 / 5.96827。
- **Transformer（實驗室伺服器 GPU）**：分支 `exp/xs-rank`，在 tmux session `xs` 依序跑
  `xformer_xs_raw`（2 層）與 `xformer_xs_raw_l5`（5 層），預計共約 2.5 小時。
  比較基準：零均值 + 3 種子 2 層（舊特徵）CV 6.14511；四模型 ensemble + 指數中性 6.13233。
