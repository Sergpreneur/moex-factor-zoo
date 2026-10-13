"""Pre-specified descriptive extensions (Fama–MacBeth, power) and clearly labelled exploratory
analyses (bootstrap-benchmark DSR, short-term reversal by turnover, EB without clusters).

Outputs: results/fm_univariate.csv, results/fm_multivariate.csv, results/power.csv,
results/exploratory_dsr_boot.csv, results/exploratory_str_turnover.csv, results/exploratory_eb_nocluster.csv
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
from moexzoo import config as C  # noqa: E402
from moexzoo import chars as CH  # noqa: E402
from moexzoo import inference as I  # noqa: E402
from moexzoo import portfolios as PF  # noqa: E402

K = len(CH.CHAR_IDS)


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    return p


def in_sample(idx: pd.PeriodIndex) -> np.ndarray:
    return (idx + 1 >= pd.Period(C.SAMPLE_START, "M")) & (idx + 1 <= pd.Period(C.SAMPLE_END, "M"))


# ----------------------------------------------------------------------------------
def fama_macbeth(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_u, rows_m = [], []
    for t, g in panel.groupby("month", sort=True):
        if not in_sample(pd.PeriodIndex([t]))[0]:
            continue
        g = g[g["U100"]]
        y = g["R_next"].fillna(0.0).values
        R = {}
        for k in CH.CHAR_IDS:
            x = g[k].where(g[k] > 0) if k == "DY" else g[k]
            r = x.rank(method="average")
            n = r.notna().sum()
            z = (r - 1) / (n - 1) - 0.5 if n > 1 else r * np.nan
            R[k] = CH.SIGN[k] * z
        Z = pd.DataFrame(R, index=g.index)
        uni = {}
        for k in CH.CHAR_IDS:
            ok = Z[k].notna().values
            if ok.sum() < C.MIN_STOCKS_PER_SORT:
                uni[k] = np.nan
                continue
            X = np.column_stack([np.ones(ok.sum()), Z[k].values[ok]])
            b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
            uni[k] = b[1]
        rows_u.append(pd.Series(uni, name=t + 1))
        Xm = np.column_stack([np.ones(len(Z)), Z.fillna(0.0).values])
        bm = np.linalg.lstsq(Xm, y, rcond=None)[0]
        rows_m.append(pd.Series(bm[1:], index=CH.CHAR_IDS, name=t + 1))
    U = pd.DataFrame(rows_u)
    M = pd.DataFrame(rows_m)
    return I.tstats(U), I.tstats(M)


# ----------------------------------------------------------------------------------
def power_study(F: pd.DataFrame, n_outer: int = 300, B_inner: int = 499, seed: int = 7) -> pd.DataFrame:
    """Single true factor among 24 nulls; common outer resamples reused across factors and levels."""
    X0 = F.dropna().values
    X0 = X0 - X0.mean(0)
    T0, Kf = X0.shape
    T = 177
    sd = X0.std(0, ddof=1)
    levels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    rng = np.random.default_rng(seed)
    # paths of T = 177 months drawn from the T0 months in which all 25 factors are observed
    outer = I.stationary_indices(T0, n_outer, C.BOOT_MEAN_BLOCK, rng, length=T)
    L = I.nw_lag(T)
    rej = {(lv, rung): np.zeros(Kf) for lv in levels for rung in ("R1", "R2", "R3", "R4", "R5", "R6")}
    for o in range(n_outer):
        Xs = X0[outer[o]]
        mean, se, _ = I._nw_mean_se(Xs, L)
        t_base, tb = I.bootstrap_tstats(Xs, B=B_inner, mean_block=C.BOOT_MEAN_BLOCK, seed=10_000 + o, L=L)
        ab = np.abs(np.nan_to_num(tb))
        for lv in levels:
            delta = lv / np.sqrt(12.0) * sd
            for j in range(Kf):
                t = t_base.copy()
                t[j] = (mean[j] + delta[j]) / se[j]
                p = 2 * stats.norm.sf(np.abs(t))
                pos = t[j] > 0
                rej[(lv, "R1")][j] += (t[j] > 1.96)
                rej[(lv, "R2")][j] += (t[j] > 3.0)
                rej[(lv, "R3")][j] += I.holm(p)[j] and pos
                rej[(lv, "R4")][j] += I.bh(p)[j] and pos
                rej[(lv, "R5")][j] += I.bh(p, dependence_free=True)[j] and pos
                # RW adjusted p for factor j (stepdown): max over steps up to j's position
                a = np.abs(t)
                order = np.argsort(-a)
                padj = 0.0
                for i, k in enumerate(order):
                    rem = order[i:]
                    pk = (1.0 + np.sum(ab[:, rem].max(1) >= a[k])) / (B_inner + 1.0)
                    padj = max(padj, pk)
                    if k == j:
                        break
                rej[(lv, "R6")][j] += (padj <= C.ALPHA) and pos
    rows = []
    for (lv, rung), v in rej.items():
        rows.append({"sr_ann": lv, "rung": rung, "power_mean": float(np.mean(v / n_outer)),
                     "power_min": float(np.min(v / n_outer)), "power_max": float(np.max(v / n_outer))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
def dsr_bootstrap_benchmark(F: pd.DataFrame, B: int = C.BOOT_REPS) -> pd.DataFrame:
    X = F.values.astype(float)
    T, Kf = X.shape
    rng = np.random.default_rng(C.BOOT_SEED + 1)
    idx = I.stationary_indices(T, B, C.BOOT_MEAN_BLOCK, rng)
    m = np.nanmean(X, 0)
    mx = []
    for a in range(0, B, 500):
        Xb = X[idx[a: a + 500]]
        mb = np.nanmean(Xb, 1)
        sb = np.nanstd(Xb, 1, ddof=1)
        mx.append(np.nanmax((mb - m) / sb, 1))
    sr0 = float(np.mean(np.concatenate(mx)))
    ts = I.tstats(F)
    d = {k: I.psr(r["sr_m"], sr0, int(r["n"]), r["skew"], r["kurt"]) for k, r in ts.iterrows()}
    out = pd.DataFrame({"sr_m": ts["sr_m"], "dsr_boot": pd.Series(d)})
    out.attrs["sr0"] = sr0
    return out, sr0


# ----------------------------------------------------------------------------------
def str_by_turnover(panel: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for t, g in panel.groupby("month", sort=True):
        if not in_sample(pd.PeriodIndex([t]))[0]:
            continue
        g = g[g["UALL"]].dropna(subset=["STR", "TURN"]).set_index("secid")
        if len(g) < 60:
            continue
        tg = PF._groups(g["TURN"], 3)
        res = {}
        for q in (1, 2, 3):
            sub = g.loc[tg.index[tg == q]]
            sg = PF._groups(sub["STR"], 5)
            hi = sub.loc[sg.index[sg == 5], "R_next"].fillna(0).mean()
            lo = sub.loc[sg.index[sg == 1], "R_next"].fillna(0).mean()
            res[f"turn{q}"] = hi - lo  # winners minus losers of month t (continuation > 0)
        rows[t + 1] = res
    D = pd.DataFrame(rows).T
    D.index = pd.PeriodIndex(D.index, freq="M")
    return I.tstats(D)


def main() -> None:
    t0 = time.time()
    panel = load_panel()
    F = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")
    F.index = pd.PeriodIndex(F.index, freq="M")
    F = F[CH.CHAR_IDS]

    fu, fm = fama_macbeth(panel)
    fu.to_csv(C.RESULTS / "fm_univariate.csv")
    fm.to_csv(C.RESULTS / "fm_multivariate.csv")
    print("FM done", f"{time.time() - t0:.0f}s")

    dsrb, sr0 = dsr_bootstrap_benchmark(F)
    dsrb.to_csv(C.RESULTS / "exploratory_dsr_boot.csv")
    print("bootstrap E[max SR] monthly", round(sr0, 4), "pass", int((dsrb["dsr_boot"] > 0.95).sum()))

    st = str_by_turnover(panel)
    st.to_csv(C.RESULTS / "exploratory_str_turnover.csv")
    print(st[["mean", "t", "n"]])

    # EB without clusters (single cluster) — exploratory sensitivity
    Tn = F.notna().sum().values
    S = F.cov(min_periods=24).values
    Sigma = I.nearest_psd(S / np.sqrt(np.outer(Tn, Tn)))
    eb1 = I.empirical_bayes(F.mean().values, Sigma, ["all"] * K)
    pd.DataFrame({"post_mean": eb1["post_mean"], "lo95": eb1["lo95"], "hi95": eb1["hi95"], "p_pos": eb1["p_pos"]},
                 index=CH.CHAR_IDS).to_csv(C.RESULTS / "exploratory_eb_nocluster.csv")
    ex = {"dsr_boot_sr0_monthly": sr0, "dsr_boot_pass": int((dsrb["dsr_boot"] > 0.95).sum()),
          "eb_nocluster": {"mu0": eb1["mu0"], "tau": float(np.hypot(eb1["tau_c"], eb1["tau_w"])),
                           "n_pos": int((eb1["lo95"] > 0).sum()), "n_neg": int((eb1["hi95"] < 0).sum())}}
    print("EB no-cluster", ex["eb_nocluster"])

    pw = power_study(F)
    pw.to_csv(C.RESULTS / "power.csv", index=False)
    print(pw.pivot(index="sr_ann", columns="rung", values="power_mean").round(3))
    (C.RESULTS / "summary_extensions.json").write_text(json.dumps(ex, indent=1, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
