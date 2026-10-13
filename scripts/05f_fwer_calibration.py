"""Exploratory (revision): family-wise error rates under the global null and a size-calibrated Romano–Wolf test.

Enlarges the simulation of 05c_mechanism.py from 1,000 to 4,000 replications in each of the same two designs
(multivariate-t(5) innovations with AR(1) coefficient 0.1 and the factors' correlation matrix; stationary
bootstrap of the demeaned factor returns), each replication applying the Romano–Wolf test with B = 499. For
every replication it stores the first-step bootstrap p-value of the largest |t| (under the global null a
family-wise error occurs exactly when this p-value is at or below the nominal level) and whether the
t > 3 rule, Bonferroni and Holm with normal p-values reject at least one hypothesis.

The calibrated level alpha* is the largest nominal level whose simulated family-wise error rate does not
exceed 5% in a design; the Romano–Wolf test is then re-run on the actual data (B = 10,000) at alpha*.

Output: results/summary_fwer.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402
from moexzoo import inference as I  # noqa: E402

REPS = 4000
B_IN = 499


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def one_rep(Y: np.ndarray, seed: int) -> dict:
    t_, tb_ = I.bootstrap_tstats(Y, B=B_IN, mean_block=C.BOOT_MEAN_BLOCK, seed=seed, chunk=B_IN)
    a = np.abs(t_)
    mx = np.nanmax(np.abs(np.nan_to_num(tb_)), 1)
    p_first = (1.0 + np.sum(mx >= np.nanmax(a))) / (B_IN + 1.0)
    p = 2 * stats.norm.sf(a)
    K = len(a)
    return {"p_first": p_first, "t3": bool((a > 3.0).any()), "bonf": bool((p <= C.ALPHA / K).any()),
            "holm": bool(I.holm(p).any()), "rw05": bool(I.romano_wolf(t_, tb_)["reject"].any())}


def summarise(recs: list[dict]) -> dict:
    n = len(recs)
    pf = np.array([r["p_first"] for r in recs])
    out = {"reps": n}
    for key in ("t3", "bonf", "holm", "rw05"):
        k = int(sum(r[key] for r in recs))
        out[key] = {"rate": k / n, "ci": wilson(k, n)}
    k = int((pf <= C.ALPHA).sum())
    out["rw_rate_05"] = {"rate": k / n, "ci": wilson(k, n)}
    grid = np.arange(1, B_IN + 2) / (B_IN + 1.0)
    rates = np.array([(pf <= g).mean() for g in grid])
    ok = grid[rates <= C.ALPHA]
    out["alpha_star"] = float(ok.max()) if len(ok) else float(grid[0])
    out["rate_at_alpha_star"] = float((pf <= out["alpha_star"]).mean())
    out["rate_10"] = float((pf <= 0.10).mean())
    return out


def main():
    t0 = time.time()
    F = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")
    F.index = pd.PeriodIndex(F.index, freq="M")
    F = F[CH.CHAR_IDS]
    T, K = F.shape
    corr = np.corrcoef(np.nan_to_num(F.fillna(F.mean()).values), rowvar=False)
    Lc = np.linalg.cholesky(I.nearest_psd(corr, 1e-8))
    res = {}

    rng = np.random.default_rng(11)
    recs = []
    for r in range(REPS):
        nu = 5
        g = rng.standard_normal((T + 50, K)) @ Lc.T
        w = np.sqrt(rng.chisquare(nu, size=(T + 50, 1)) / nu)
        e = g / w
        Y = np.empty_like(e)
        Y[0] = e[0]
        for s in range(1, T + 50):
            Y[s] = 0.1 * Y[s - 1] + e[s]
        recs.append(one_rep(Y[50:], 50_000 + r))
        if (r + 1) % 1000 == 0:
            print("parametric", r + 1, f"{time.time() - t0:.0f}s", flush=True)
    res["parametric"] = summarise(recs)

    X0 = F.dropna().values
    X0 = X0 - X0.mean(0)
    idx = I.stationary_indices(len(X0), REPS, C.BOOT_MEAN_BLOCK, np.random.default_rng(12), length=T)
    recs = []
    for r in range(REPS):
        recs.append(one_rep(X0[idx[r]], 60_000 + r))
        if (r + 1) % 1000 == 0:
            print("bootstrap", r + 1, f"{time.time() - t0:.0f}s", flush=True)
    res["bootstrap"] = summarise(recs)

    # calibrated Romano–Wolf on the actual data
    X = F.values.astype(float)
    t_obs, tb = I.bootstrap_tstats(X, B=C.BOOT_REPS, mean_block=C.BOOT_MEAN_BLOCK)
    mx = np.nanmax(np.abs(np.nan_to_num(tb)), 1)
    a_star = min(res["parametric"]["alpha_star"], res["bootstrap"]["alpha_star"])
    rw = I.romano_wolf(t_obs, tb, alpha=a_star)
    res["calibrated"] = {"alpha_star": a_star, "crit_first": float(np.quantile(mx, 1 - a_star)),
                         "survivors": [k for k, rej, t in zip(CH.CHAR_IDS, rw["reject"], t_obs) if rej and t > 0],
                         "wrong": [k for k, rej, t in zip(CH.CHAR_IDS, rw["reject"], t_obs) if rej and t < 0],
                         "p_rw": {k: float(p) for k, p in zip(CH.CHAR_IDS, rw["padj"])}}
    (C.RESULTS / "summary_fwer.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
