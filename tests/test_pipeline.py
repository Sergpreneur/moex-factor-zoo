"""Integrity tests: no look-ahead in characteristics, correct corporate-action handling,
correct alignment of holding returns. Run: python3 -m pytest -q tests"""

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


@pytest.fixture(scope="module")
def P():
    return CH.Panel()


@pytest.mark.parametrize("t", ["2014-03", "2018-06", "2022-02", "2024-11"])
def test_no_lookahead(P, t):
    t = pd.Period(t, "M")
    base = CH.characteristics_at(P, t)
    Q = copy.copy(P)
    end = np.where(P.month_of == t)[0][-1]
    rng = np.random.default_rng(0)
    after = np.arange(len(P.sessions)) > end
    for name in ("close", "high", "low", "value", "ret", "div", "tri", "own_chg"):
        df = getattr(P, name).copy()
        noise = rng.uniform(0.5, 1.5, size=df.shape)
        df.loc[after, :] = df.loc[after, :].values * noise[after]
        setattr(Q, name, df)
    Q.traded = P.traded.copy()
    Q.traded.loc[after, :] = rng.uniform(size=(after.sum(), P.traded.shape[1])) > 0.5
    Q.mret = P.mret.copy()
    Q.mret[after] = rng.normal(0, 0.05, after.sum())
    Q.mret_lag = Q.mret.shift(1)
    later = P.R.index > t
    for name in ("R", "value_m", "cap"):
        df = getattr(P, name).copy()
        df.loc[later, :] = rng.normal(0, 1, size=(later.sum(), df.shape[1]))
        setattr(Q, name, df)
    Q.mkt_m = P.mkt_m.copy()
    Q.mkt_m[Q.mkt_m.index > t] = 0.3
    pert = CH.characteristics_at(Q, t)
    pd.testing.assert_frame_equal(base, pert, check_exact=False, rtol=1e-12, atol=1e-12)


def test_split_continuity(P):
    # GMKN 1:100 split on 2024-04-08 and VTBR 5000:1 reverse split on 2024-07-15
    for sec, day in (("GMKN", "2024-04-08"), ("VTBR", "2024-07-15"), ("TRNFP", "2024-02-21")):
        r = P.ret[sec].loc[pd.Timestamp(day) - pd.Timedelta(days=7): pd.Timestamp(day) + pd.Timedelta(days=7)]
        assert r.abs().max() < 0.25, (sec, r.abs().max())


def test_dividend_on_ex_date(P):
    # SBER record date 2024-07-11 (T+1 regime) -> ex-date 2024-07-11, 33.3 RUB
    d = P.div["SBER"]
    assert abs(d.loc["2024-07-11"] - 33.3) < 1e-9
    assert d.loc["2024-07-01":"2024-07-31"].sum() == pytest.approx(33.3)
    # total return that day = (P + D)/P_prev - 1 ; price return is about -10%
    r, px = P.ret["SBER"].loc["2024-07-11"], P.close["SBER"]
    pr = px.loc["2024-07-11"] / px.loc["2024-07-10"] - 1
    assert r == pytest.approx(pr + 33.3 / px.loc["2024-07-10"], rel=1e-9)


def test_holding_return_alignment():
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    m = pd.read_parquet(C.INTERIM / "monthly.parquet")
    R = m.pivot(index="month", columns="secid", values="ret")
    R.index = pd.PeriodIndex(R.index, freq="M")
    sub = panel[(panel["secid"] == "SBER")].set_index("month")
    for t in [pd.Period("2015-05", "M"), pd.Period("2023-12", "M")]:
        assert sub.loc[t, "R_next"] == pytest.approx(R.loc[t + 1, "SBER"])
