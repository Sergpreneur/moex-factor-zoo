"""Exploratory: decomposition of the Romano–Wolf first-step critical value (95% quantile of the maximum of 25
absolute t-statistics under the null) into the effects of (a) correlation among the factors, (b) finite-sample
noise in the Newey–West standard error, and (c) the remaining features of the actual returns preserved by the
bootstrap. Adds keys to results/summary_mechanism.json and saves results/mech_iid_nw_max.npy (independent case,
L = 4) used in Figure 3.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402
from moexzoo import inference as I  # noqa: E402


def nw_t(Y: np.ndarray, L: int) -> np.ndarray:
    m, s, _ = I._nw_mean_se(Y, L)
    return m / s


def main():
    t0 = time.time()
    M = json.loads((C.RESULTS / "summary_mechanism.json").read_text())
    F = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")[CH.CHAR_IDS]
    corr = I.nearest_psd(F.corr().values, 1e-10)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    Lc = np.linalg.cholesky(corr)
    T, K = 177, 25
    q = 1 - C.ALPHA
    rng = np.random.default_rng(2026)

    # (a) Gaussian statistics: independent vs factor correlation
    Z = rng.standard_normal((1_000_000, K))
    M["q_normal_indep"] = float(np.quantile(np.abs(Z).max(1), q))
    zc = np.abs(Z @ Lc.T).max(1)
    M["q_normal_corr"] = float(np.quantile(zc, q))
    np.save(C.RESULTS / "mech_corr_normal_max.npy", zc[:200_000])

    # (b) Newey–West t from i.i.d. normal returns, T = 177: independent factors (1,000,000 series -> 40,000 maxima)
    res = {L: [] for L in (4, 8, 12)}
    p309 = []
    for _ in range(20):
        Y = rng.standard_normal((50_000, T, 1))
        for L in res:
            t = nw_t(Y, L)[:, 0]
            res[L].append(t)
            if L == 4:
                p309.append(np.abs(t) > 3.09)
    for L in res:
        tt = np.concatenate(res[L])
        mx = np.abs(tt.reshape(-1, K)).max(1)
        M[f"q_iidnw_indep_L{L}"] = float(np.quantile(mx, q))
        if L == 4:
            np.save(C.RESULTS / "mech_iid_nw_max.npy", mx)
            M["iid_nw_sd"] = float(tt.std())
    M["iid_nw_p309"] = float(np.concatenate(p309).mean())
    M["iid_nw_max_crit"] = M["q_iidnw_indep_L4"]
    print("indep done", f"{time.time() - t0:.0f}s", flush=True)

    # (b') the same with the factors' correlation matrix (100,000 replications)
    mxs = {L: [] for L in (4, 8, 12)}
    for _ in range(25):
        E = rng.standard_normal((4_000, T, K)) @ Lc.T
        for L in mxs:
            mxs[L].append(np.abs(nw_t(E, L)).max(1))
    for L in mxs:
        M[f"q_iidnw_corr_L{L}"] = float(np.quantile(np.concatenate(mxs[L]), q))
    np.save(C.RESULTS / "mech_iid_nw_corr_max.npy", np.concatenate(mxs[4]))
    M["crit_L4"] = M["crit_baseline"]
    (C.RESULTS / "summary_mechanism.json").write_text(json.dumps(M, indent=1, default=float))
    keys = ["q_normal_indep", "q_normal_corr", "q_iidnw_indep_L4", "q_iidnw_corr_L4", "crit_baseline",
            "q_iidnw_indep_L8", "q_iidnw_corr_L8", "crit_L8", "q_iidnw_indep_L12", "q_iidnw_corr_L12", "crit_L12",
            "iid_nw_p309", "iid_nw_sd", "sidak_crit_neff15", "crit_rescaled_unit_sd", "crit_normal_scores"]
    for k in keys:
        print(f"{k:22s} {M[k]:.4f}")
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
