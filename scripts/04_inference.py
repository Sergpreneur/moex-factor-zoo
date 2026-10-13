"""Run the inference ladder (plan §6) on the baseline and on every robustness variant (§7),
plus N_eff, PBO, empirical Bayes, regimes (H4) and costs (H5).

Outputs: results/ladder_<variant>.csv, results/summary.json, results/eb_baseline.csv,
results/regimes.csv, results/costs.csv, results/pbo_logits.npy, results/boot_baseline.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402
from moexzoo import chars as CH  # noqa: E402
from moexzoo import inference as I  # noqa: E402

FAC = C.RESULTS / "factors"
K = len(CH.CHAR_IDS)
N_CONFIGS = K * 10


def load(v: str) -> pd.DataFrame:
    F = pd.read_parquet(FAC / f"{v}.parquet")
    F.index = pd.PeriodIndex(F.index, freq="M")
    return F[CH.CHAR_IDS]


def market_excess() -> pd.Series:
    mm = pd.read_parquet(C.INTERIM / "market_monthly.parquet")
    mm.index = pd.PeriodIndex(mm.index, freq="M")
    return mm["mkt_ret"] - mm["rf"]


def ladder(F: pd.DataFrame, *, block: float = C.BOOT_MEAN_BLOCK, B: int = C.BOOT_REPS, dsr_N: float = K,
           tag: str = "") -> tuple[pd.DataFrame, dict]:
    ts = I.tstats(F)
    p = ts["p"].values
    pos = ts["t"].values > 0
    t_obs, tb = I.bootstrap_tstats(F.values.astype(float), B=B, mean_block=block)
    rw = I.romano_wolf(t_obs, tb)
    dsr, sr0 = I.deflated_sharpe(ts, N=dsr_N)
    out = ts.copy()
    out["R1_t196"] = ts["t"] > 1.96
    out["R2_hlz3"] = ts["t"] > C.HLZ_HURDLE
    out["R3_holm"] = I.holm(p) & pos
    out["R4_bh"] = I.bh(p) & pos
    out["R5_by"] = I.bh(p, dependence_free=True) & pos
    out["p_bh"] = I.bh_adjusted(p)
    out["p_by"] = I.bh_adjusted(p, dependence_free=True)
    out["p_rw"] = rw["padj"]
    out["R6_rw"] = rw["reject"] & pos
    out["dsr"] = dsr.reindex(out.index).values
    out["R7_dsr"] = out["dsr"] > C.DSR_PASS
    out["wrong_sign_rw"] = rw["reject"] & ~pos
    out["wrong_sign_t196"] = ts["t"] < -1.96
    info = {"rw_crit_first": rw["crit_first"], "rw_crit_steps": rw["crit_steps"], "dsr_sr0_monthly": sr0,
            "global_null_p": float((1 + np.sum(np.nanmax(np.abs(np.nan_to_num(tb)), 1) >= np.nanmax(np.abs(t_obs)))) / (B + 1)),
            "counts": {c: int(out[c].sum()) for c in ["R1_t196", "R2_hlz3", "R3_holm", "R4_bh", "R5_by", "R6_rw", "R7_dsr",
                                                       "wrong_sign_rw", "wrong_sign_t196"]}}
    return out, info


def main() -> None:
    summary: dict = {}
    mx = market_excess()

    # ---------------- baseline -----------------------------------------------------
    F = load("baseline")
    lad, info = ladder(F)
    lad.to_csv(C.RESULTS / "ladder_baseline.csv")
    summary["baseline"] = info
    print("baseline counts", info["counts"], "RW crit", round(info["rw_crit_first"], 3))

    # block length sensitivity
    for b in (3, 12):
        l2, i2 = ladder(F, block=b)
        summary[f"baseline_block{b}"] = {"counts": i2["counts"], "rw_crit_first": i2["rw_crit_first"]}
        l2[["t", "p_rw", "R6_rw"]].to_csv(C.RESULTS / f"ladder_baseline_block{b}.csv")

    # effective number of tests
    corr = F.corr().values
    ne = I.neff(corr)
    summary["neff"] = {"nyholt": ne["nyholt"], "li_ji": ne["li_ji"], "eigen_top5": ne["eigen"][:5].tolist(),
                       "corr_mean_abs_offdiag": float(np.mean(np.abs(corr[np.triu_indices(K, 1)])))}
    ts = I.tstats(F)
    dsr_neff, sr0_neff = I.deflated_sharpe(ts, N=ne["li_ji"])
    dsr_all, sr0_all = I.deflated_sharpe(ts, N=N_CONFIGS)
    lad["dsr_Neff"] = dsr_neff.reindex(lad.index).values
    lad["dsr_N250"] = dsr_all.reindex(lad.index).values
    summary["dsr"] = {"sr0_N25": info["dsr_sr0_monthly"], "sr0_Neff": sr0_neff, "sr0_N250": sr0_all,
                      "pass_N25": int((lad["dsr"] > C.DSR_PASS).sum()), "pass_Neff": int((lad["dsr_Neff"] > C.DSR_PASS).sum()),
                      "pass_N250": int((lad["dsr_N250"] > C.DSR_PASS).sum())}
    lad.to_csv(C.RESULTS / "ladder_baseline.csv")

    # PBO
    pb = I.pbo_cscv(F)
    np.save(C.RESULTS / "pbo_logits.npy", pb["logits"])
    is_best = {}
    summary["pbo"] = {"pbo": pb["pbo"], "T_common": pb["T"], "K": pb["K"]}
    print("N_eff", summary["neff"], "PBO", summary["pbo"])

    # bootstrap draws for figures (max-|t| distribution)
    t_obs, tb = I.bootstrap_tstats(F.values.astype(float), B=C.BOOT_REPS)
    np.savez_compressed(C.RESULTS / "boot_baseline.npz", t_obs=t_obs, maxabs=np.nanmax(np.abs(np.nan_to_num(tb)), 1))

    # empirical Bayes on oriented mean returns
    Xc = F.dropna()
    Tn = F.notna().sum().values
    S = F.cov(min_periods=24).values  # pairwise
    Sigma = I.nearest_psd(S / np.sqrt(np.outer(Tn, Tn)))
    eb = I.empirical_bayes(F.mean().values, Sigma, [CH.CLUSTER[k] for k in CH.CHAR_IDS])
    ebdf = pd.DataFrame({"mean": F.mean().values, "post_mean": eb["post_mean"], "post_sd": eb["post_sd"],
                         "lo95": eb["lo95"], "hi95": eb["hi95"], "p_pos": eb["p_pos"],
                         "cluster": [CH.CLUSTER[k] for k in CH.CHAR_IDS]}, index=CH.CHAR_IDS)
    ebdf.to_csv(C.RESULTS / "eb_baseline.csv")
    summary["eb"] = {"mu0": eb["mu0"], "tau_c": eb["tau_c"], "tau_w": eb["tau_w"],
                     "n_replicate_pos": int((ebdf["lo95"] > 0).sum()), "n_negative": int((ebdf["hi95"] < 0).sum()),
                     "replication_rate": float((ebdf["lo95"] > 0).mean())}
    print("EB", summary["eb"])

    # ---------------- CAPM alpha (market-hedged series) ---------------------------------
    alph = {}
    hedged = {}
    for k in CH.CHAR_IDS:
        y = F[k]
        x = mx.reindex(y.index)
        d = pd.concat([y, x], axis=1).dropna()
        L = I.nw_lag(len(d))
        res = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
        alph[k] = {"alpha": res.params.iloc[0], "t_alpha": res.tvalues.iloc[0], "beta_mkt": res.params.iloc[1],
                   "t_beta": res.tvalues.iloc[1]}
        hedged[k] = y - res.params.iloc[1] * x
    H = pd.DataFrame(hedged)[CH.CHAR_IDS]
    lad_c, info_c = ladder(H)
    A = pd.DataFrame(alph).T
    lad_c = lad_c.join(A)
    lad_c.to_csv(C.RESULTS / "ladder_capm.csv")
    summary["capm"] = info_c

    # ---------------- variants -------------------------------------------------------
    for v in ["vw", "u50", "uall", "terciles", "lag1", "u100ord", "priceonly", "extrans", "pit"]:
        Fv = load(v)
        lv, iv = ladder(Fv)
        lv.to_csv(C.RESULTS / f"ladder_{v}.csv")
        summary[v] = iv
        print(v, iv["counts"], round(iv["rw_crit_first"], 3))

    # ---------------- regimes (H4) -----------------------------------------------------
    post = pd.Series((F.index >= pd.Period(C.REGIME_BREAK, "M")).astype(float), index=F.index)
    rows = []
    for k in CH.CHAR_IDS + ["ZOO_EW"]:
        y = F[k] if k != "ZOO_EW" else F.mean(axis=1)
        d = pd.concat([y.rename("y"), post.rename("post")], axis=1).dropna()
        L = I.nw_lag(len(d))
        res = sm.OLS(d["y"], sm.add_constant(d["post"])).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
        pre_m, post_m = d.loc[d["post"] == 0, "y"], d.loc[d["post"] == 1, "y"]
        tp = I.tstats(pd.DataFrame({"pre": pre_m.reset_index(drop=True)}))
        tq = I.tstats(pd.DataFrame({"post": post_m.reset_index(drop=True)}))
        rows.append({"factor": k, "mean_pre": pre_m.mean(), "t_pre": tp["t"].iloc[0], "T_pre": len(pre_m),
                     "mean_post": post_m.mean(), "t_post": tq["t"].iloc[0], "T_post": len(post_m),
                     "delta": res.params["post"], "t_delta": res.tvalues["post"], "p_delta": res.pvalues["post"]})
    RG = pd.DataFrame(rows).set_index("factor")
    RG["p_delta_bh"] = np.nan
    RG.loc[CH.CHAR_IDS, "p_delta_bh"] = I.bh_adjusted(RG.loc[CH.CHAR_IDS, "p_delta"].values)
    # ladders within each regime
    for name, sub in (("pre", F[post == 0]), ("post", F[post == 1])):
        ls, isub = ladder(sub)
        ls.to_csv(C.RESULTS / f"ladder_{name}.csv")
        summary[f"regime_{name}"] = isub
    RG.to_csv(C.RESULTS / "regimes.csv")
    summary["regimes"] = {"n_bh_sig": int((RG.loc[CH.CHAR_IDS, "p_delta_bh"] < 0.05).sum()),
                          "zoo_delta": float(RG.loc["ZOO_EW", "delta"]), "zoo_t_delta": float(RG.loc["ZOO_EW", "t_delta"])}
    print("regimes", summary["regimes"])

    # ---------------- costs (H5) ---------------------------------------------------------
    diag = pd.read_parquet(FAC / "diagnostics_baseline.parquet")
    Nn = load("baseline_net")
    N2 = load("baseline_net2x")
    tn, t2 = I.tstats(Nn), I.tstats(N2)
    to = diag.groupby("factor")["turnover"].mean()
    cst = diag.groupby("factor")["cost"].mean()
    nl = diag.groupby("factor")[["n_long", "n_short"]].mean()
    ncash = diag.groupby("factor")["n_cash"].sum()
    CO = pd.DataFrame({"gross_mean": F.mean(), "net_mean": tn["mean"], "net_t": tn["t"], "net2x_mean": t2["mean"],
                       "net2x_t": t2["t"], "turnover_2sided": to, "avg_cost": cst,
                       "breakeven_oneway_bp": 1e4 * F.mean() / to, "n_long": nl["n_long"], "n_short": nl["n_short"],
                       "cash_names_total": ncash}).reindex(CH.CHAR_IDS)
    lad_n, info_n = ladder(Nn)
    CO["net_R6_rw"] = lad_n["R6_rw"]
    CO.to_csv(C.RESULTS / "costs.csv")
    summary["costs"] = info_n
    (C.RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
