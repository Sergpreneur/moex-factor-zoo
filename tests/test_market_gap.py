"""The daily market return must not span the 2022 trading suspension (pre-specified 14-day rule)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402


def test_market_return_not_spanning_suspension():
    m = pd.read_parquet(C.INTERIM / "market_daily.parquet")
    assert np.isnan(m.loc["2022-03-25", "ret"])
    assert np.isnan(m.loc["2022-03-24", "ret"])
    assert abs(m.loc["2022-03-28", "ret"]) < 0.1
