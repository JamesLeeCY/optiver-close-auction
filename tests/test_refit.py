import numpy as np
import pytest

from src.train_lgb import refit_schedule


@pytest.mark.parametrize("gap", [0, 2])
def test_refit_only_uses_dates_before_each_chunk(gap):
    train, emb, val = np.arange(0, 30), np.arange(30, 35), np.arange(35, 58)
    sched = refit_schedule(train, val, every_days=5, gap_days=gap, embargo_dates=emb)
    # 每段預測的日期剛好覆蓋整個驗證期、不重疊
    assert np.array_equal(np.concatenate([p for _, p in sched]), val)
    assert np.array_equal(sched[0][0], train)
    for fit, pred in sched:
        assert set(train) <= set(fit)
        seen = np.setdiff1d(fit, train)
        # 只用這段開始前、再往前空 gap 天的日期
        assert len(seen) == 0 or seen.max() < pred.min() - gap
        assert np.array_equal(seen, np.concatenate([emb, val])[:len(seen)])
    # 第二段起包含 embargo 那幾天；gap=0 時最後一段可用到它前一天為止的所有日期
    assert set(emb) <= set(sched[1][0])
    if gap == 0:
        assert np.setdiff1d(sched[-1][0], train).max() == sched[-1][1].min() - 1
