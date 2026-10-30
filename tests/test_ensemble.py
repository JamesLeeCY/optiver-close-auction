import numpy as np
import pandas as pd

from src.ensemble import neutralize


def test_neutralize_removes_index_mean():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"stock_id": np.tile([0, 1, 2], 4),
                      "date_id": np.repeat([0, 0, 1, 1], 3),
                      "seconds_in_bucket": np.repeat([0, 10, 0, 10], 3)}).iloc[:-1]  # 最後一個時點缺一檔
    w = pd.Series([0.5, 0.3, 0.2])
    p = rng.normal(size=len(d))
    q = neutralize(d, p, w)
    wi = d["stock_id"].map(w).to_numpy()
    keys = d["date_id"] * 100 + d["seconds_in_bucket"]
    assert np.allclose(pd.Series(wi * q).groupby(keys.to_numpy()).sum(), 0.0)
    # 每個時點只是平移，同一時點內的相對大小不變
    assert np.allclose(np.diff(q[:3]), np.diff(p[:3]))
