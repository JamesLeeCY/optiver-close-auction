import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic import make  # noqa: E402

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    df, w = make(n_stocks=12, n_days=20, seed=1)
    df = df.drop(columns=["row_id", "time_id"])
    df = df.sort_values(["date_id", "seconds_in_bucket", "stock_id"]).reset_index(drop=True)
    return df, w


@pytest.fixture(scope="session")
def fcfg():
    return load_config(ROOT / "configs" / "base.yaml")["features"]
