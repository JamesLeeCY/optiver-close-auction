# Transformer 輸出零均值：中途暫停的紀錄（2026-11-06）

改動：`XStockNet(zero_mean=True)` 在每個時點把輸出減掉真實股票（不含補位）的平均，參考 Kaggle 第一名
（hyd）的 `out = out - out.mean(1, keepdim=True)`。其他設定與 `configs/xformer.yaml` 相同（單一種子 42）。

本機跑到 fold 3 第 7 個 epoch 時手動暫停，改到伺服器上重跑。前 3 折已完成：

| fold | 基準（xformer，單種子） | zero_mean | 差 |
| --- | --- | --- | --- |
| 0 | 6.25119 | 6.25027（epoch 5） | −0.0009 |
| 1 | 5.90500 | 5.90813（epoch 5） | +0.0031 |
| 2 | 6.39848 | 6.39788（epoch 2） | −0.0006 |
| 3 | 6.26618 | 未完成（epoch 0–7 最佳 val 約 6.272） | — |
| 4 | 5.97497 | 未跑 | — |

前 3 折平均差 +0.0005，目前看不出改善。完整 log 見 `train_partial.log`。

重跑指令（6 執行緒，單折約 15–20 分鐘，記憶體峰值約 18 GB）：

    OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1 python -m src.train_xformer --config configs/xformer_zeromean.yaml
