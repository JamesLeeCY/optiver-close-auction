# Optiver Trading at the Close — Nasdaq 收盤競價價格預測

> **一句話摘要**：以 purged walk-forward CV 訓練 LightGBM + MLP ensemble，預測個股 WAP 未來 60 秒相對合成指數的變動（bps）。
> CV MAE **[待填]** · Private LB **[待填]**（第一名約 **[待填]**）。

<!-- 前 30 秒要讓面試官看到結果表與一張圖：完成後把 reports/ 的圖放這裡 -->

## 1. 問題背景

Nasdaq 每天 15:50–16:00 進行 Closing Cross：交易所每 10 秒公布一次競價資訊（已配對量、未配對量 imbalance、參考價、far / near price），最後以單一價格撮合收盤單。收盤價是指數基金與衍生品結算的基準，收盤前 10 分鐘的價格發現對做市商很重要。本題要預測每支股票 WAP 未來 60 秒相對加權指數的超額報酬，以 MAE 評分。

## 2. 驗證設計

```
dates:  |=============== CV 可用日期 (90%) ===============|== holdout (10%) ==|
fold 0: [train ............]  gap  [val]
fold 1: [train ..................]  gap  [val]
fold 2: [train ........................]  gap  [val]
...                              (gap = embargo_days)
```

- **以 `date_id` 切分**，不隨機切：同一天不同股票的資料共享同一個市場狀態。
- **Walk-forward expanding window**：每折用之前所有日期訓練，驗證接下來的一段。
- **Embargo**：訓練集最後一天與驗證集之間空 `embargo_days` 天。target 只看 60 秒、不跨日，理論上不需要太長，但這是好習慣。
- **Holdout**：最後 10% 日期完全不碰，`--holdout` 只跑一次，紀錄在 `reports/holdout_log.csv`（重複使用會跳警告）。
- **指數權重**只用第一折的訓練日期估計（它是所有折訓練集的子集）。
- **洩漏測試**（`tests/test_leakage.py`，CI 自動跑）：竄改未來秒數或其他天的原始資料，檢查過去的特徵完全不變。

## 3. 結果

| 模型 | CV MAE | Private LB | 訓練時間 |
| --- | --- | --- | --- |
| 預測 0 | [待填] | — | — |
| LightGBM | [待填] | [待填] | [待填] |
| MLP | [待填] | — | [待填] |
| Ensemble（nested 權重） | [待填] | [待填] | — |

兩模型 OOF 相關係數：[待填]（`reports/ensemble_report.json`）

## 4. 哪些特徵有效、為什麼

| 特徵組 | 例子 | 市場理由 | 拿掉後 ΔMAE |
| --- | --- | --- | --- |
| basic | spread、book imbalance | 掛單量不對稱反映短期買賣壓力 | [待填] |
| auction | imbalance / matched、帶方向 imbalance | 未配對量會推動最終收盤價 | [待填] |
| time | seconds_in_bucket、300 秒旗標 | 300 秒後才有 far / near price | [待填] |
| temporal | lag、diff、rolling mean（只往前看） | 失衡在擴大還是收斂 | [待填] |
| cross_sectional | 減去同時點平均、排名 | target 是相對指數 | [待填] |
| index | 個股報酬減加權指數報酬 | 直接對應 target 定義 | [待填] |
| stock | 個股 target 波動度（只從訓練折算） | 不同股票雜訊尺度不同 | [待填] |

數字來自 `python -m src.analyze ablation`；SHAP 圖見 `reports/shap_summary.png`。

## 5. 沒用的嘗試

1. [待填：做了什麼、結果、你認為失敗的原因]
2. [待填]
3. [待填]

## 6. 限制與下一步

- 沒有考慮交易成本與市場衝擊，MAE 好不等於能賺錢
- 只預測 60 秒，不處理最終收盤價
- 尚未實作 online 更新（測試期會揭露前一期 target，可增量更新 LightGBM）
- Transformer（以同時點所有股票為 token 學橫截面）留作延伸

## 7. 如何重現

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 資料：先在 Kaggle 比賽頁同意規則，並設定 ~/.kaggle/kaggle.json
bash scripts/download_data.sh

pytest -q tests                                       # 洩漏與 CV 測試
python -m src.train_lgb --config configs/base.yaml    # LightGBM baseline
python -m src.train_nn  --config configs/base.yaml    # MLP
python -m src.ensemble  --config configs/base.yaml --models lgb,nn
python -m src.analyze ablation --config configs/base.yaml
python -m src.analyze shap     --config configs/base.yaml
python -m src.train_lgb --config configs/base.yaml --holdout   # 只跑一次
```

沒有資料時可先用合成資料跑通：

```bash
python scripts/make_synthetic.py --stocks 20 --days 60
python -m src.train_lgb --config configs/smoke.yaml
```

開發時在 `configs/base.yaml` 設 `data.max_dates` / `max_stocks` 縮小資料。預計執行時間：[待填]。

## Repo 結構

```
configs/   base.yaml（正式）、smoke.yaml（合成資料）
src/       data · cv · features · index_weights · pipeline · train_lgb · train_nn · ensemble · analyze
scripts/   download_data.sh、make_synthetic.py
tests/     CV 切分、洩漏、指數權重估計
reports/   CV log、OOF、ablation、SHAP
```
