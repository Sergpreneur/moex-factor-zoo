"""Inference tests: NW against statsmodels, BH/BY/Holm against statsmodels, Romano–Wolf FWER
control under a correlated, fat-tailed global null, DSR monotonicity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import inference as I  # noqa: E402


def test_nw_matches_statsmodels():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=300)) * 0.01 + rng.normal(size=300)
    L = 4
    mean, se, n = I._nw_mean_se(x[:, None], L)
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
    assert np.isclose(se[0], res.bse[0], rtol=1e-10)
    assert np.isclose(mean[0], res.params[0])


def test_multiplicity_matches_statsmodels():
    rng = np.random.default_rng(2)
    p = np.sort(rng.uniform(0, 0.2, 25))
    assert (I.holm(p) == multipletests(p, 0.05, "holm")[0]).all()
    assert (I.bh(p) == multipletests(p, 0.05, "fdr_bh")[0]).all()
    assert (I.bh(p, dependence_free=True) == multipletests(p, 0.05, "fdr_by")[0]).all()
    assert np.allclose(I.bh_adjusted(p), multipletests(p, 0.05, "fdr_bh")[1])


def test_romano_wolf_controls_fwer():
    rng = np.random.default_rng(3)
    K, T, sims = 12, 177, 120
    A = rng.normal(size=(K, K))
    cov = A @ A.T / K + np.eye(K) * 0.5
    Lc = np.linalg.cholesky(cov)
    false = 0
    for s in range(sims):
        e = rng.standard_t(5, size=(T, K)) @ Lc.T
        # AR(1) dependence
        X = np.empty_like(e)
        X[0] = e[0]
        for t in range(1, T):
            X[t] = 0.2 * X[t - 1] + e[t]
        t_obs, tb = I.bootstrap_tstats(X, B=399, mean_block=6, seed=s, chunk=400)
        rw = I.romano_wolf(t_obs, tb)
        false += rw["reject"].any()
    rate = false / sims
    assert rate <= 0.10, rate  # nominal 5%; generous bound for 120 replications


def test_dsr_monotone_in_trials():
    ts = pd.DataFrame({"sr_m": [0.3, 0.1, 0.0, -0.1, 0.2], "n": [177] * 5, "skew": [0] * 5, "kurt": [3] * 5})
    d1, s1 = I.deflated_sharpe(ts, N=2)
    d2, s2 = I.deflated_sharpe(ts, N=50)
    assert s2 > s1
    assert (d2 <= d1 + 1e-12).all()
