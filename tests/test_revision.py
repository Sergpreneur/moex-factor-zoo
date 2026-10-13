"""Tests for the helpers of the revision analyses (moexzoo/revision.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402
from moexzoo import portfolios as PF  # noqa: E402
from moexzoo import revision as RV  # noqa: E402


def test_ols_multi_masked_matches_lstsq():
    rng = np.random.default_rng(1)
    T, N = 60, 7
    X = rng.normal(size=(T, 2))
    X[5, 1] = np.nan
    y = 0.3 + (X @ np.array([1.5, -0.7]))[:, None] + rng.normal(scale=0.1, size=(T, N)) + rng.normal(size=(1, N))
    y[rng.uniform(size=(T, N)) < 0.15] = np.nan
    coef, resid, n = RV.ols_multi_masked(y, X)
    for j in range(N):
        ok = ~np.isnan(y[:, j]) & ~np.isnan(X).any(1)
        Z = np.column_stack([np.ones(ok.sum()), X[ok]])
        b = np.linalg.lstsq(Z, y[ok, j], rcond=None)[0]
        assert np.allclose(coef[j], b, atol=1e-10)
        assert n[j] == ok.sum()
        assert np.allclose(resid[ok, j], y[ok, j] - Z @ b, atol=1e-10)
        assert np.isnan(resid[~ok, j]).all()


@pytest.fixture(scope="module")
def panel():
    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    return p


@pytest.mark.parametrize("k", ["HIGH52", "STR", "DY"])
def test_ls_ext_reproduces_baseline(panel, k):
    a = RV.ls_ext(panel, k, CH.SIGN[k])
    b = PF.long_short(panel, k, CH.SIGN[k])
    idx = a.index.intersection(b.ret.index)
    assert np.allclose(a.loc[idx, "ret"], b.ret.loc[idx], equal_nan=True, atol=1e-12)
    assert np.allclose(a.loc[idx, "turnover"], b.turnover.loc[idx], equal_nan=True, atol=1e-12)
    assert np.allclose(a.loc[idx, "cost"], b.cost.loc[idx], equal_nan=True, atol=1e-12)


def test_chars_2f_use_no_factor_data_after_t():
    P = CH.Panel()
    U = pd.read_parquet(C.PROCESSED / "universes.parquet")
    smb_m, smb_d = RV.build_smb(P, U)
    t = pd.Period("2019-06", "M")
    c0 = RV.chars_2f(P, t, smb_m, smb_d)
    rng = np.random.default_rng(0)
    m2 = smb_m.copy()
    m2[m2.index > t] = rng.normal(size=int((m2.index > t).sum()))
    d2 = smb_d.copy()
    after = P.month_of > t
    d2[after] = rng.normal(size=int(after.sum()))
    c1 = RV.chars_2f(P, t, m2, d2)
    pd.testing.assert_frame_equal(c0, c1)
