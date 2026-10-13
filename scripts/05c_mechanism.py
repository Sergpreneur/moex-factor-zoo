"""Exploratory analyses added after the results were seen (listed in prereg/deviations.md):

1. What drives the Romano–Wolf critical value: dispersion and kurtosis of the bootstrap statistics, a
   normal-score transform of the factor returns, rescaling of t* to unit variance, i.i.d.-normal benchmark
   for Newey–West t-statistics, longer HAC lags, and Monte Carlo error across seeds.
2. A stronger simulation check of family-wise error control by the Romano–Wolf implementation.
3. PBO details: which factor is the in-sample winner, and its out-of-sample rank.
4. Turnover composition of U100 and short-term returns by turnover within U100.
5. Empirical Bayes with the nine cluster labels of the plan's Table (sensitivity).
6. Mean market excess return over the sample.

Outputs: results/summary_mechanism.json, results/mech_iid_nw_max.npy
"""

from __future__ import annotations

import itertools
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
from moexzoo import portfolios as PF  # noqa: E402

OUT: dict = {}


def crit_of(X, L=4, block=C.BOOT_MEAN_BLOCK, B=C.BOOT_REPS, seed=C.BOOT_SEED):
    t, tb = I.bootstrap_tstats(X, B=B, mean_block=block, seed=seed, L=L)
    mx = np.nanmax(np.abs(np.nan_to_num(tb)), 1)
    return float(np.quantile(mx, 1 - C.ALPHA)), t, tb


def main():
    t0 = time.time()
    F = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")
    F.index = pd.PeriodIndex(F.index, freq="M")
    F = F[CH.CHAR_IDS]
    X = F.values.astype(float)
    T, K = X.shape

    # ---------------- 1. mechanism ------------------------------------------------
    c0, t_obs, tb = crit_of(X)
    OUT["crit_baseline"] = c0
    sd = np.nanstd(tb, 0)
    ku = np.array([stats.kurtosis(tb[:, k][~np.isnan(tb[:, k])], fisher=False) for k in range(K)])
    OUT["tstar_sd_median"], OUT["tstar_sd_min"], OUT["tstar_sd_max"] = float(np.median(sd)), float(sd.min()), float(sd.max())
    OUT["tstar_kurt_median"], OUT["tstar_kurt_max"] = float(np.median(ku)), float(ku.max())
    tbs = tb / sd
    OUT["crit_rescaled_unit_sd"] = float(np.quantile(np.nanmax(np.abs(np.nan_to_num(tbs)), 1), 1 - C.ALPHA))
    Z = np.full_like(X, np.nan)
    kret = []
    for k in range(K):
        x = X[:, k]
        ok = ~np.isnan(x)
        kret.append(stats.kurtosis(x[ok], fisher=False))
        r = stats.rankdata(x[ok])
        Z[ok, k] = stats.norm.ppf((r - 0.5) / ok.sum())
    OUT["return_kurt_median"] = float(np.median(kret))
    OUT["crit_normal_scores"] = crit_of(Z)[0]
    # i.i.d. normal returns, T=177, NW lag 4
    rng = np.random.default_rng(0)
    tt = []
    for _ in range(10):
        Y = rng.standard_normal((20000, T, 1))
        m, s, _n = I._nw_mean_se(Y, I.nw_lag(T))
        tt.append((m / s)[:, 0])
    tt = np.concatenate(tt)
    OUT["iid_nw_p309"] = float((np.abs(tt) > 3.09).mean())
    OUT["iid_nw_p196"] = float((np.abs(tt) > 1.96).mean())
    OUT["iid_nw_sd"] = float(tt.std())
    mx = np.abs(tt.reshape(-1, K)).max(1)
    np.save(C.RESULTS / "mech_iid_nw_max.npy", mx)
    OUT["iid_nw_max_crit"] = float(np.quantile(mx, 1 - C.ALPHA))
    OUT["sidak_crit_neff15"] = float(stats.norm.ppf(1 - (1 - (1 - C.ALPHA) ** (1 / 15)) / 2))
    OUT["sidak_crit_25"] = float(stats.norm.ppf(1 - (1 - (1 - C.ALPHA) ** (1 / 25)) / 2))
    for L in (8, 12):
        cL, tL, _ = crit_of(X, L=L)
        OUT[f"crit_L{L}"] = cL
        OUT[f"tratio_L{L}"] = float(np.nanmean(np.abs(tL) / np.abs(t_obs)))
    ac = np.array([[pd.Series(X[:, k]).autocorr(l) for l in range(1, 13)] for k in range(K)])
    OUT["share_pos_autocorr_sum12"] = float(np.mean(np.nansum(ac, 1) > 0))
    OUT["mean_autocorr_lag1"] = float(np.nanmean(ac[:, 0]))
    # Monte Carlo error across seeds
    seeds = {}
    for sd_ in (1, 2, 3, 4, 5):
        t_, tb_ = I.bootstrap_tstats(X, B=C.BOOT_REPS, mean_block=C.BOOT_MEAN_BLOCK, seed=sd_)
        rw = I.romano_wolf(t_, tb_)
        surv = sorted(np.array(CH.CHAR_IDS)[rw["reject"] & (t_ > 0)].tolist())
        wrong = sorted(np.array(CH.CHAR_IDS)[rw["reject"] & (t_ < 0)].tolist())
        seeds[sd_] = {"crit": rw["crit_first"], "survivors": surv, "wrong": wrong,
                      "p_abvol": float(rw["padj"][CH.CHAR_IDS.index("ABVOL")]),
                      "p_mom127": float(rw["padj"][CH.CHAR_IDS.index("MOM12_7")])}
    OUT["seeds"] = seeds
    print("mechanism done", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 2. FWER simulation ---------------------------------------------
    corr = np.corrcoef(np.nan_to_num(F.fillna(F.mean()).values), rowvar=False)
    Lc = np.linalg.cholesky(I.nearest_psd(corr, 1e-8))
    rng = np.random.default_rng(11)
    reps = 1000
    fw_par = 0
    for r in range(reps):
        nu = 5
        g = rng.standard_normal((T + 50, K)) @ Lc.T
        w = np.sqrt(rng.chisquare(nu, size=(T + 50, 1)) / nu)
        e = g / w
        Y = np.empty_like(e)
        Y[0] = e[0]
        for s in range(1, T + 50):
            Y[s] = 0.1 * Y[s - 1] + e[s]
        Y = Y[50:]
        t_, tb_ = I.bootstrap_tstats(Y, B=499, mean_block=C.BOOT_MEAN_BLOCK, seed=50_000 + r, chunk=499)
        fw_par += bool(I.romano_wolf(t_, tb_)["reject"].any())
    OUT["fwer_sim_reps"] = reps
    OUT["fwer_sim_parametric"] = fw_par / reps
    # nonparametric: stationary bootstrap of demeaned factors (global null)
    X0 = F.dropna().values
    X0 = X0 - X0.mean(0)
    idx = I.stationary_indices(len(X0), reps, C.BOOT_MEAN_BLOCK, np.random.default_rng(12), length=T)
    fw_np = 0
    for r in range(reps):
        Y = X0[idx[r]]
        t_, tb_ = I.bootstrap_tstats(Y, B=499, mean_block=C.BOOT_MEAN_BLOCK, seed=60_000 + r, chunk=499)
        fw_np += bool(I.romano_wolf(t_, tb_)["reject"].any())
    OUT["fwer_sim_nonparametric"] = fw_np / reps
    print("FWER sims done", OUT["fwer_sim_parametric"], OUT["fwer_sim_nonparametric"], f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 3. PBO details ------------------------------------------------
    M = F.dropna()
    A = M.values
    Tm = len(A)
    blocks = np.array_split(np.arange(Tm), C.CSCV_SPLITS)
    best_names, oos_ranks = [], []
    for comb in itertools.combinations(range(C.CSCV_SPLITS), C.CSCV_SPLITS // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(C.CSCV_SPLITS) if i not in comb])
        sr_is = A[is_idx].mean(0) / A[is_idx].std(0, ddof=1)
        sr_oos = A[oos_idx].mean(0) / A[oos_idx].std(0, ddof=1)
        b = int(np.argmax(sr_is))
        best_names.append(M.columns[b])
        oos_ranks.append(stats.rankdata(sr_oos)[b])
    best_names = pd.Series(best_names)
    oos_ranks = np.array(oos_ranks)
    mom = {"MOM12_1", "MOM6_1", "MOM12_7", "RESMOM", "HIGH52"}
    OUT["pbo_splits"] = len(best_names)
    OUT["pbo_share_momentum_best"] = float(best_names.isin(mom).mean())
    OUT["pbo_share_high52_best"] = float((best_names == "HIGH52").mean())
    OUT["pbo_share_abvol_best"] = float((best_names == "ABVOL").mean())
    OUT["pbo_oos_rank_median"] = float(np.median(oos_ranks))
    OUT["pbo_oos_top5_share"] = float(np.mean(oos_ranks >= K - 4))
    OUT["pbo_oos_top1_share"] = float(np.mean(oos_ranks == K))

    # ---------------- 4. turnover: U100 composition and STR within U100 ----------------
    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    p = p[(p["month"] + 1 >= pd.Period(C.SAMPLE_START, "M")) & (p["month"] + 1 <= pd.Period(C.SAMPLE_END, "M"))]
    shares_top = []
    rows = {}
    for t, g in p.groupby("month", sort=True):
        ga = g[g["UALL"]].dropna(subset=["TURN"]).set_index("secid")
        if len(ga) >= 60:
            tg = PF._groups(ga["TURN"], 3)
            u = ga[ga["U100"]]
            shares_top.append(float((tg.reindex(u.index) == 3).mean()))
        gu = g[g["U100"]].dropna(subset=["TURN", "STR"]).set_index("secid")
        if len(gu) >= 50:
            th = PF._groups(gu["TURN"], 2)
            res = {}
            for q in (1, 2):
                sub = gu.loc[th.index[th == q]]
                sg = PF._groups(sub["STR"], 5)
                res[q] = sub.loc[sg.index[sg == 5], "R_next"].fillna(0).mean() - sub.loc[sg.index[sg == 1], "R_next"].fillna(0).mean()
            rows[t + 1] = {"u100_low_turn": res[1], "u100_high_turn": res[2], "u100_high_minus_low": res[2] - res[1]}
    OUT["u100_share_in_top_turn_tercile"] = float(np.mean(shares_top))
    D = pd.DataFrame(rows).T
    D.index = pd.PeriodIndex(D.index, freq="M")
    ts = I.tstats(D)
    for k in ts.index:
        OUT[k + "_mean"] = float(ts.loc[k, "mean"])
        OUT[k + "_t"] = float(ts.loc[k, "t"])

    # ---------------- 5. EB with nine clusters ---------------------------------------
    nine = {"MOM12_1": "Momentum", "MOM6_1": "Momentum", "MOM12_7": "Momentum", "RESMOM": "Momentum",
            "HIGH52": "Momentum", "STR": "Reversal", "LTR": "Reversal", "SEAS": "Seasonality", "BETA": "Low risk",
            "IVOL": "Low risk", "TVOL": "Low risk", "MAX": "Low risk", "DBETA": "Downside", "SKEW": "Skewness",
            "ISKEW": "Skewness", "COSKEW": "Skewness", "SIZE": "Size", "AMIHUD": "Liquidity", "DVOL": "Liquidity",
            "TURN": "Liquidity", "ZERO": "Liquidity", "SPREAD": "Liquidity", "CVVAL": "Liquidity",
            "ABVOL": "Liquidity", "DY": "Payout"}
    Tn = F.notna().sum().values
    S = F.cov(min_periods=24).values
    Sigma = I.nearest_psd(S / np.sqrt(np.outer(Tn, Tn)))
    eb9 = I.empirical_bayes(F.mean().values, Sigma, [nine[k] for k in CH.CHAR_IDS])
    pos9 = [k for k, lo in zip(CH.CHAR_IDS, eb9["lo95"]) if lo > 0]
    OUT["eb9_n_pos"] = len(pos9)
    OUT["eb9_pos"] = pos9
    OUT["eb9_tau_c"], OUT["eb9_tau_w"] = eb9["tau_c"], eb9["tau_w"]
    nc = pd.read_csv(C.RESULTS / "exploratory_eb_nocluster.csv", index_col=0)
    OUT["eb_nocluster_pos"] = [k for k in CH.CHAR_IDS if nc.loc[k, "lo95"] > 0]
    ebc = pd.read_csv(C.RESULTS / "eb_baseline.csv", index_col=0)
    OUT["eb_cluster_pos"] = [k for k in CH.CHAR_IDS if ebc.loc[k, "lo95"] > 0]
    OUT["eb_post_sd_median"] = float(ebc["post_sd"].median())
    ts0 = I.tstats(F)
    OUT["ols_se_median"] = float(ts0["se"].median())
    # error correlations quoted in the text
    cm = F.corr()
    OUT["corr_ivol_tvol"] = float(cm.loc["IVOL", "TVOL"])
    OUT["corr_mom121_mom127"] = float(cm.loc["MOM12_1", "MOM12_7"])

    # ---------------- 6. market excess return ------------------------------------------
    mm = pd.read_parquet(C.INTERIM / "market_monthly.parquet")
    mm.index = pd.PeriodIndex(mm.index, freq="M")
    mm = mm[(mm.index >= pd.Period(C.SAMPLE_START, "M")) & (mm.index <= pd.Period(C.SAMPLE_END, "M"))]
    OUT["mkt_excess_mean"] = float((mm["mkt_ret"] - mm["rf"]).mean())

    (C.RESULTS / "summary_mechanism.json").write_text(json.dumps(OUT, indent=1, default=float))
    print(json.dumps({k: v for k, v in OUT.items() if k != "seeds"}, indent=0, default=float)[:3000])
    print("seeds", json.dumps(seeds, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
