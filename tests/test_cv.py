import numpy as np
import pytest

from src.cv import cv_setup, split_holdout, walk_forward_folds

DATES = np.repeat(np.arange(481), 3)  # 真實資料有 481 天，每天多列


def test_val_strictly_after_train_with_embargo():
    for f in walk_forward_folds(DATES, n_folds=5, embargo_days=5):
        assert f.train_dates.max() + 5 < f.val_dates.min()


def test_expanding_and_val_blocks_disjoint_contiguous():
    folds = walk_forward_folds(DATES, n_folds=5, embargo_days=5)
    assert len(folds) == 5
    for a, b in zip(folds, folds[1:]):
        assert len(b.train_dates) > len(a.train_dates)
        assert a.val_dates.max() < b.val_dates.min()
    for f in folds:
        assert np.all(np.diff(f.val_dates) == 1)


def test_holdout_is_last_and_untouched():
    folds, holdout = cv_setup(DATES, {"n_folds": 5, "embargo_days": 5, "holdout_frac": 0.1})
    assert len(holdout) == 48
    used = np.concatenate([np.concatenate([f.train_dates, f.val_dates]) for f in folds])
    assert used.max() < holdout.min()


def test_no_holdout():
    cv_dates, hold = split_holdout(DATES, 0.0)
    assert len(hold) == 0 and len(cv_dates) == 481


def test_too_small_raises():
    with pytest.raises(ValueError):
        walk_forward_folds(np.arange(6), n_folds=3, embargo_days=5)
