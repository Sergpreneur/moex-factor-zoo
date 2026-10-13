"""Look-ahead and data-construction checks added in the revision: universe membership, trading-cost estimates
and characteristics at month t must not depend on data after t; adjusted share counts must be continuous
through splits."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402
from moexzoo import data as D  # noqa: E402
from moexzoo import universe as UN  # noqa: E402


@pytest.fixture(scope="module")
def monthly_sessions():
    m = pd.read_parquet(C.INTERIM / "monthly.parquet")
    sessions = pd.read_parquet(C.INTERIM / "daily_close.parquet").index
    return m, sessions


@pytest.mark.parametrize("t", ["2013-06", "2017-11", "2022-02", "2025-03"])
def test_universe_uses_no_data_after_t(monthly_sessions, t):
    m, sessions = monthly_sessions
    t = pd.Period(t, "M")
    base = UN.build(m, sessions)
    rng = np.random.default_rng(1)
    q = m.copy()
    after = pd.PeriodIndex(q["month"], freq="M") > t
    for col in ("value_m", "ndays", "gap_to_end"):
        q[col] = q[col].astype(float)
        q.loc[after, col] = np.round(q.loc[after, col].values * rng.uniform(0.0, 3.0, after.sum()))
    pert = UN.build(q, sessions)
    cols = ["secid", "liq_rank", "UALL", "U100", "U50", "U100ORD"]
    a = base[base["month"] == t][cols].sort_values("secid").reset_index(drop=True)
    b = pert[pert["month"] == t][cols].sort_values("secid").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


@pytest.fixture(scope="module")
def P():
    return CH.Panel()


@pytest.mark.parametrize("t", ["2014-03", "2022-02", "2024-11"])
def test_costs_use_no_data_after_t(P, t):
    t = pd.Period(t, "M")
    base = CH.costs_at(P, t)
    Q = copy.copy(P)
    end = np.where(P.month_of == t)[0][-1]
    after = np.arange(len(P.sessions)) > end
    rng = np.random.default_rng(2)
    for name in ("high", "low", "close"):
        df = getattr(P, name).copy()
        df.loc[after, :] = df.loc[after, :].values * rng.uniform(0.5, 1.5, size=(after.sum(), df.shape[1]))
        setattr(Q, name, df)
    pd.testing.assert_series_equal(base, CH.costs_at(Q, t))


def test_share_counts_continuous_through_splits():
    m = pd.read_parquet(C.INTERIM / "monthly.parquet")
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    checked = 0
    for f in (C.MM / "splits").glob("*.csv"):
        s = f.stem
        g = m[m["secid"] == s].set_index("month").sort_index()
        for d, _ in D.load_splits(s):
            before, after = g[g.index < d.to_period("M")], g[g.index >= d.to_period("M")]
            if before.empty or after.empty:
                continue
            b, a = before.iloc[-1], after.iloc[0]
            if pd.isna(b["shares_adj"]) or pd.isna(a["shares_adj"]):
                continue
            assert abs(a["shares_adj"] / b["shares_adj"] - 1.0) < 1e-9, (s, d)
            assert abs((a["cap"] / b["cap"]) / (a["close"] / b["close"]) - 1.0) < 1e-9, (s, d)
            checked += 1
    assert checked >= 10
    # no month-to-month jump in adjusted share counts larger than a factor of three anywhere
    sh = m.pivot(index="month", columns="secid", values="shares_adj").sort_index()
    r = (sh / sh.shift(1)).stack()
    assert ((r > 3) | (r < 1 / 3)).sum() == 0
