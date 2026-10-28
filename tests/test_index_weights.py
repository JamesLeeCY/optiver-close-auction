import numpy as np

from src.index_weights import estimate_index_weights


def test_recovers_true_weights(synth):
    df, true_w = synth
    est = estimate_index_weights(df).reindex(true_w.index)
    assert np.allclose(est.sum(), 1.0)
    assert np.abs(est - true_w).max() < 0.01
