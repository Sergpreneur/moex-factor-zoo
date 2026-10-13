"""Exploratory analyses added in the revision, in response to external comments on the first version
(prereg/deviations.md, section D; labelled as exploratory in the paper). None of them changes a
pre-specified result.

1. Terminal events (delistings) and suspensions: exposure of the portfolio legs, bounds with assumed
   delisting returns, and a hold-to-resumption treatment of suspended shares.
2. Depositary-receipt histories: U100 rebuilt without the six issuers whose receipts appear in the data.
3. RESMOM and IVOL with a two-factor (market + SMB) model instead of the CAPM.
4. Implementability of the survivors: long and short legs, borrow fees, the 2022 short-sale ban,
   short legs restricted to liquid or index shares, and an illustrative square-root price-impact charge.
5. Confidence intervals and minimum detectable effects for the 2022 regime differences.
6. Redundancy among the survivors: correlations and spanning regressions.

Outputs: results/summary_revision.json and results/revision_*.csv / .parquet
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402
from moexzoo import data as D  # noqa: E402
from moexzoo import inference as I  # noqa: E402
from moexzoo import portfolios as PF  # noqa: E402
from moexzoo import revision as RV  # noqa: E402
from moexzoo import universe as UN  # noqa: E402

_spec = importlib.util.spec_from_file_location("inf04", Path(__file__).resolve().parent / "04_inference.py")
inf04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf04)

OUT: dict = {}
SURV = ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1", "ABVOL"]
MOMS = ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1"]
DR_ISSUERS = ["TCSG", "OZON", "VKCO", "ETLN", "MDMG", "GEMC"]  # receipt histories in the source data
# holding months whose short legs would have been opened while the ban (24 Feb - 31 May 2022) was in force:
# formation at the closes of February, March, April and May 2022
BAN_MONTHS = tuple(pd.Period(m, "M") for m in ("2022-03", "2022-04", "2022-05", "2022-06"))
AUM_REF = 1e9  # impact is proportional to impact_y * sqrt(AUM), so one run at AUM_REF gives every size exactly


def in_sample(s):
    return s[(s.index >= pd.Period(C.SAMPLE_START, "M")) & (s.index <= pd.Period(C.SAMPLE_END, "M"))]


def tt(s: pd.Series) -> tuple[float, float]:
    r = I.tstats(pd.DataFrame({"x": s}))
    return float(r["mean"].iloc[0]), float(r["t"].iloc[0])


def run_ladder(F: pd.DataFrame) -> dict:
    lad, info = inf04.ladder(F[CH.CHAR_IDS])
    surv = [k for k in CH.CHAR_IDS if lad.loc[k, "R6_rw"]]
    return {"counts": info["counts"], "crit": info["rw_crit_first"], "survivors": surv,
            "t": {k: float(lad.loc[k, "t"]) for k in CH.CHAR_IDS}, "p_rw": {k: float(lad.loc[k, "p_rw"]) for k in CH.CHAR_IDS},
            "wrong_rw": [k for k in CH.CHAR_IDS if lad.loc[k, "wrong_sign_rw"]]}


def factors_from(panel, *, cols=None, **kw) -> pd.DataFrame:
    out = {}
    for k in (cols or CH.CHAR_IDS):
        r = RV.ls_ext(panel, k, CH.SIGN[k], **kw)
        out[k] = in_sample(r["ret"])
    return pd.DataFrame(out)


def main() -> None:
    t0 = time.time()
    P = CH.Panel()
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    base = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")
    base.index = pd.PeriodIndex(base.index, freq="M")
    base = base[CH.CHAR_IDS]

    # ---------------- panel augmentation ------------------------------------------------
    mm = pd.read_parquet(C.INTERIM / "monthly.parquet")[["month", "secid", "value_m"]]
    mm["month"] = pd.PeriodIndex(mm["month"], freq="M")
    panel = panel.merge(mm, on=["month", "secid"], how="left")
    spm = P.sess_per_month
    panel["adv"] = panel["value_m"] / panel["month"].map(spm).astype(float)

    # terminal events: last trade of the security at least five sessions before the end of the data, and
    # the security is not the predecessor in an ISS ticker change
    traded = P.traded
    last_trade = traded.apply(lambda c: c.index[c.values].max() if c.any() else pd.NaT)
    end_cut = P.sessions[-6]
    tick = json.loads((C.MM / "tickers.json").read_text())
    preds = {p for p, s, r in UN.rename_edges(tick)}
    lt = pd.DatetimeIndex(last_trade.reindex(panel["secid"]).values)
    lt_m = pd.PeriodIndex(lt, freq="M")
    hm = panel["month"] + 1
    dead = (lt < end_cut) & ~panel["secid"].isin(preds).values
    panel["term_partial"] = dead & (lt_m == hm)
    panel["term_none"] = dead & (lt_m < hm)
    panel["terminal"] = panel["term_partial"] | panel["term_none"]
    # each terminal event is charged once, to the U100 portfolios that held the share when it stopped trading:
    # in the holding month that contains the last trade; if the share is still in U100 at the next formation
    # date (and so held, as cash, in the following month), that month is not charged again. A share that enters
    # U100 only in its last trading month is charged in the month after.
    u100 = panel["U100"].values
    tp = panel["term_partial"].values & u100
    tn = panel["term_none"].values & u100
    sec_tp = set(panel.loc[tp, "secid"])
    panel["charge_partial"] = tp
    panel["charge_none"] = tn & ~panel["secid"].isin(sec_tp).values
    panel["terminal_charged"] = panel["charge_partial"] | panel["charge_none"]
    cp, cn = panel["charge_partial"], panel["charge_none"]
    for d in (0.30, 0.50, 1.00):
        col = f"R_dl{int(d * 100)}"
        r = panel["R_next"].copy()
        r[cp] = (1 + panel.loc[cp, "R_next"].fillna(0.0)) * (1 - d) - 1
        r[cn] = -d
        panel[col] = r
    r = panel["R_next"].copy()  # buyout premium of 50% on charged events (used for short legs in a two-sided bound)
    r[cp] = (1 + panel.loc[cp, "R_next"].fillna(0.0)) * 1.5 - 1
    r[cn] = 0.5
    panel["R_up50"] = r
    # daily volatility for the impact charge: TVOL of month t (at least 15 daily returns); if missing (March 2022
    # had six sessions), the s.d. of the month's daily returns with at least five observations; then the share's
    # TVOL of the previous formation month; then the month's cross-sectional median (forward-filled)
    sd_m = P.ret.groupby(P.month_of).std(ddof=1)
    n_m = P.ret.groupby(P.month_of).count()
    sd_m = sd_m.where(n_m >= 5)
    sd_m.index = pd.PeriodIndex(sd_m.index, freq="M")
    sd_long = sd_m.stack().rename("sd_short").reset_index()
    sd_long.columns = ["month", "secid", "sd_short"]
    panel = panel.merge(sd_long, on=["month", "secid"], how="left")
    panel = panel.sort_values(["secid", "month"]).reset_index(drop=True)
    prev_m = panel.groupby("secid")["month"].shift(1)
    prev_v = panel.groupby("secid")["TVOL"].shift(1)
    consecutive = (panel["month"] - prev_m).apply(lambda x: getattr(x, "n", np.nan)) == 1
    panel["sig"] = panel["TVOL"].fillna(panel["sd_short"]).fillna(prev_v.where(consecutive))
    med = panel.groupby("month")["sig"].median().ffill()
    panel["sig"] = panel["sig"].fillna(panel["month"].map(med))
    panel = panel.sort_values(["month", "secid"]).reset_index(drop=True)
    # suspensions: no trade in the holding month, trading resumes later -> return to the first trade after
    tri = P.tri
    susp = panel["R_next"].isna() & ~panel["term_none"]
    panel["suspended"] = susp
    rhold = panel["R_next"].copy()
    susp_rows = []
    month_end = pd.Series(P.sessions, index=P.sessions).groupby(P.month_of).max()
    for i in np.where(susp.values)[0]:
        s, t = panel.at[i, "secid"], panel.at[i, "month"]
        col = tri[s]
        before = col[col.index <= month_end[t]].dropna()
        after = col[col.index > month_end[t]].dropna()
        if len(before) and len(after):
            rhold.iat[i] = after.iloc[0] / before.iloc[-1] - 1.0
            susp_rows.append({"secid": s, "formation": str(t), "resumed": str(after.index[0].date()),
                              "ret": float(rhold.iat[i]), "U100": bool(panel.at[i, "U100"])})
    panel["R_hold"] = rhold
    OUT["suspensions"] = susp_rows

    # receipt histories removed: re-rank UALL without the six issuers
    noDR = panel["UALL"] & ~panel["secid"].isin(DR_ISSUERS)
    rk = panel[noDR].groupby("month")["liq_rank"].rank(method="first")
    panel["U100_noDR"] = False
    panel.loc[rk.index, "U100_noDR"] = (rk <= C.TOP_N_BASELINE).values

    # short-eligibility proxies: U50 membership; MOEX Russia index constituent at the formation date
    bases = D.load_index_bases()
    eff = sorted(bases["effective_from"].unique())
    members = {e: set(bases.loc[bases["effective_from"] == e, "secid"].str.rstrip("*")) for e in eff}
    me = pd.DatetimeIndex(panel["month"].map(month_end).values)
    kk = np.searchsorted(np.array(eff, dtype="datetime64[ns]"), me.values.astype("datetime64[ns]"), side="right") - 1
    # no base is in force before December 2012: index eligibility is undefined there (no backfilling), and the
    # index-restricted short leg is evaluated only from formation month 2012-12 (holding months 2013:01-2026:09)
    panel["ok_U50"] = panel["U50"]
    panel["ok_IMOEX"] = [(s in members[eff[j]]) if j >= 0 else False for s, j in zip(panel["secid"], kk)]
    panel["has_index_base"] = kk >= 0
    IDX_START = panel.loc[panel["has_index_base"], "month"].min() + 1  # first holding month with a base
    OUT["imoex_first_base"] = str(pd.Timestamp(eff[0]).date())
    files = {f.stem for f in (C.MM / "prices_iss").glob("*.csv")}
    bb = bases[~bases["secid"].str.endswith("*")].copy()
    # VK's receipt history before the 2021 ticker change is stitched into VKCO, so MAIL is not missing
    bb["miss"] = ~bb["secid"].isin(files) & (bb["secid"] != "MAIL")
    gmiss = bb.groupby("effective_from")["miss"].sum()
    OUT["index_missing"] = {"max": int(gmiss.max()), "bases_with_missing": int((gmiss > 0).sum()), "bases": int(len(gmiss)),
                            "secids": sorted(bb.loc[bb["miss"], "secid"].unique().tolist())}
    print("panel augmented", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 0. the extended function reproduces the baseline -----------------
    maxdiff, maxdiff_to, maxdiff_cost = 0.0, 0.0, 0.0
    for k in CH.CHAR_IDS:
        a = RV.ls_ext(panel, k, CH.SIGN[k])
        b = PF.long_short(panel, k, CH.SIGN[k])
        maxdiff = max(maxdiff, float(np.nanmax(np.abs(in_sample(a["ret"]) - in_sample(b.ret)))))
        maxdiff_to = max(maxdiff_to, float(np.nanmax(np.abs(in_sample(a["turnover"]) - in_sample(b.turnover)))))
        maxdiff_cost = max(maxdiff_cost, float(np.nanmax(np.abs(in_sample(a["cost"]) - in_sample(b.cost)))))
        d2 = float(np.nanmax(np.abs(in_sample(a["ret"]) - base[k].reindex(in_sample(a["ret"]).index))))
        maxdiff = max(maxdiff, d2)
    OUT["check_ls_ext_maxdiff"] = {"ret": maxdiff, "turnover": maxdiff_to, "cost": maxdiff_cost}
    assert maxdiff < 1e-10 and maxdiff_to < 1e-10 and maxdiff_cost < 1e-10, OUT["check_ls_ext_maxdiff"]
    print("ls_ext reproduces baseline", OUT["check_ls_ext_maxdiff"], flush=True)

    # ---------------- 1. terminal events -------------------------------------------------
    u = panel[panel["U100"] & ((panel["month"] + 1) >= pd.Period(C.SAMPLE_START, "M"))
              & ((panel["month"] + 1) <= pd.Period(C.SAMPLE_END, "M"))]
    OUT["term"] = {"u100_stock_months": int(len(u)), "partial": int(u["term_partial"].sum()),
                   "none": int(u["term_none"].sum()), "suspended": int(u["suspended"].sum()),
                   "partial_mean_ret": float(u.loc[u["term_partial"], "R_next"].mean()),
                   "partial_below_m20": int((u.loc[u["term_partial"], "R_next"] < -0.2).sum()),
                   "secids": int(u.loc[u["terminal"], "secid"].nunique()),
                   "charged": int(u["terminal_charged"].sum()), "charged_partial": int(u["charge_partial"].sum()),
                   "charged_none": int(u["charge_none"].sum()),
                   "none_after_partial": int((u["term_none"] & ~u["charge_none"]).sum())}
    expo = {}
    for k in CH.CHAR_IDS:
        r = RV.ls_ext(panel, k, CH.SIGN[k], count_col="terminal_charged")
        r = r[(r.index >= pd.Period(C.SAMPLE_START, "M")) & (r.index <= pd.Period(C.SAMPLE_END, "M"))]
        expo[k] = {"long": int(r["flag_l"].sum()), "short": int(r["flag_s"].sum())}
    OUT["term_exposure"] = expo
    bounds = {}
    for name, kw in (("dl30", dict(ret_col="R_dl30")), ("dl50", dict(ret_col="R_dl50")),
                     ("dl100", dict(ret_col="R_dl100")),
                     ("adverse", dict(ret_col_long="R_dl100", ret_col_short="R_next")),
                     ("adverse2", dict(ret_col_long="R_dl100", ret_col_short="R_up50")),
                     ("hold", dict(ret_col="R_hold"))):
        F = factors_from(panel, **kw)
        bounds[name] = run_ladder(F)
        F.index = F.index.astype(str)
        F.to_parquet(C.RESULTS / f"revision_factors_{name}.parquet")
        print("bound", name, bounds[name]["counts"]["R6_rw"], bounds[name]["survivors"], f"{time.time() - t0:.0f}s", flush=True)
    OUT["bounds"] = bounds

    # ---------------- 2. depositary-receipt histories removed ------------------------------
    F = factors_from(panel, universe="U100_noDR")
    OUT["noDR"] = run_ladder(F)
    OUT["noDR"]["u100_dr_stock_months"] = int(u["secid"].isin(DR_ISSUERS).sum())
    OUT["noDR"]["u100_dr_max_per_month"] = int(u[u["secid"].isin(DR_ISSUERS)].groupby("month").size().max())
    F.index = F.index.astype(str)
    F.to_parquet(C.RESULTS / "revision_factors_noDR.parquet")
    print("noDR", OUT["noDR"]["counts"], OUT["noDR"]["survivors"], f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 3. two-factor RESMOM and IVOL ------------------------------------------
    U = pd.read_parquet(C.PROCESSED / "universes.parquet")
    smb_m, smb_d = RV.build_smb(P, U)
    smb_m.to_frame("smb").assign(month=lambda d: d.index.astype(str)).to_parquet(C.RESULTS / "revision_smb_monthly.parquet")
    sm_s = in_sample(smb_m)
    OUT["smb"] = {"mean": float(sm_s.mean()), "t": tt(sm_s)[1], "corr_mkt": float(np.corrcoef(
        sm_s.values, (P.mkt_m - P.rf_m).reindex(sm_s.index).values)[0, 1]), "n_months_total": int(smb_m.notna().sum()),
        "first": str(smb_m.index.min())}
    rows = []
    for t in sorted(panel["month"].unique()):
        c2 = RV.chars_2f(P, t, smb_m, smb_d)
        c2["month"] = t
        rows.append(c2.reset_index(names="secid"))
    c2 = pd.concat(rows, ignore_index=True)
    panel = panel.merge(c2, on=["month", "secid"], how="left")
    # cross-sectional rank correlation of the CAPM and two-factor versions within U100
    rc = {}
    for a, b in (("RESMOM", "RESMOM2F"), ("IVOL", "IVOL2F")):
        cs = panel[panel["U100"]].groupby("month").apply(lambda g: g[a].corr(g[b], method="spearman"))
        rc[a] = float(cs.mean())
    f2 = {}
    for k, kk in (("RESMOM", "RESMOM2F"), ("IVOL", "IVOL2F")):
        r = in_sample(RV.ls_ext(panel, kk, CH.SIGN[k])["ret"])
        f2[k] = r
    F2 = base.copy()
    for k in f2:
        F2[k] = f2[k].reindex(F2.index)
    lad2 = run_ladder(F2)
    OUT["twofactor"] = {"rank_corr": rc, "ladder": lad2,
                        "t2f": {k: tt(f2[k])[1] for k in f2}, "mean2f": {k: tt(f2[k])[0] for k in f2},
                        "corr_factor": {k: float(np.corrcoef(f2[k].dropna(), base[k].reindex(f2[k].dropna().index))[0, 1]) for k in f2}}
    print("two-factor", OUT["twofactor"]["t2f"], lad2["survivors"], f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 4. implementability -------------------------------------------------------
    imp = {}
    for k in SURV + ["STR"]:
        sign = CH.SIGN[k] if k != "STR" else +1  # STR row: the continuation strategy (winners minus losers)
        r = RV.ls_ext(panel, k, sign)
        r = r[(r.index >= pd.Period(C.SAMPLE_START, "M")) & (r.index <= pd.Period(C.SAMPLE_END, "M"))]
        gross = r["ret"]
        net = r["ret"] - r["cost"]
        long_ex = r["long"] - r["bench"]
        short_ex = r["bench"] - r["short"]
        long_net = r["long"] - r["cost_l"] - r["bench"]
        short_net = r["bench"] - r["short"] - r["cost_s"]
        row = {"gross": tt(gross), "net": tt(net), "long_ex": tt(long_ex), "short_ex": tt(short_ex),
               "long_ex_net": tt(long_net), "short_ex_net": tt(short_net),
               "share_long": float(long_ex.mean() / gross.mean()),
               "be_borrow_pa": float(12 * net.mean()), "to_l": float(r["to_l"].mean()), "to_s": float(r["to_s"].mean())}
        for f in (0.05, 0.10, 0.20):
            row[f"net_fee{int(f * 100)}"] = tt(net - f / 12)
        for mode in ("longonly", "flat"):
            rb = RV.ls_ext(panel, k, sign, ban_months=BAN_MONTHS, ban_mode=mode)
            rb = rb[(rb.index >= pd.Period(C.SAMPLE_START, "M")) & (rb.index <= pd.Period(C.SAMPLE_END, "M"))]
            row[f"ban_{mode}_gross"] = tt(rb["ret"])
            row[f"ban_{mode}_net"] = tt(rb["ret"] - rb["cost"])
        for okc in ("ok_U50", "ok_IMOEX"):
            rs = RV.ls_ext(panel, k, sign, short_ok_col=okc)
            start = pd.Period(C.SAMPLE_START, "M") if okc == "ok_U50" else IDX_START
            rs = rs[(rs.index >= start) & (rs.index <= pd.Period(C.SAMPLE_END, "M"))]
            row[f"{okc}_gross"] = tt(rs["ret"])
            row[f"{okc}_net"] = tt(rs["ret"] - rs["cost"])
            row[f"{okc}_fallback_months"] = int(rs["fallback"].astype(bool).sum())
            row[f"{okc}_n_short"] = float(rs["n_s"].mean())
            row[f"{okc}_start"] = str(start)
        sub = net[net.index >= IDX_START]
        row["net_idx_window"] = tt(sub)  # unrestricted net over the window of the index-restricted test
        post0 = pd.Period(C.REGIME_BREAK, "M")
        ra = RV.ls_ext(panel, k, sign, aum=AUM_REF, impact_y=1.0, vol_col="sig")
        ra = ra[(ra.index >= pd.Period(C.SAMPLE_START, "M")) & (ra.index <= pd.Period(C.SAMPLE_END, "M"))]
        i1 = ra["impact"] / np.sqrt(AUM_REF)  # impact per unit of sqrt(AUM) with impact_y = 1
        nsp = ra["ret"] - ra["cost"]
        cap = {}
        for a in (1e8, 1e9):
            nn = nsp - np.sqrt(a) * i1
            cap[str(a)] = {"net": tt(nn), "net_post": tt(nn[nn.index >= post0]), "impact": float((np.sqrt(a) * i1).mean()),
                           "impact_post": float((np.sqrt(a) * i1)[i1.index >= post0].mean())}
        row["capacity"] = cap
        for yv, tag in ((1.0, ""), (0.5, "_half")):
            for sub, stag in ((slice(None), ""), (nsp.index >= post0, "_post")):
                m, ii = float(nsp[sub].mean()), float(yv * i1[sub].mean())
                row["be_aum" + stag + tag] = float((m / ii) ** 2) if m > 0 else 0.0
        row["impact_zero_months"] = int(((ra["turnover"] > 0) & (ra["impact"] <= 0)).sum())
        imp[k] = row
        print("implementability", k, f"{time.time() - t0:.0f}s", flush=True)
    OUT["implement"] = imp
    # median ADV and daily vol of U100 names (for the text)
    uu = panel[panel["U100"]]
    OUT["adv_median_by_year"] = {str(y): float(g["adv"].median()) for y, g in uu.groupby(uu["month"].dt.year)}

    # ---------------- 5. regime confidence intervals --------------------------------------------
    RG = pd.read_csv(C.RESULTS / "regimes.csv", index_col=0)
    RG["se"] = RG["delta"] / RG["t_delta"]
    RG["lo95"] = RG["delta"] - 1.96 * RG["se"]
    RG["hi95"] = RG["delta"] + 1.96 * RG["se"]
    RG["mde80"] = (1.959964 + 0.841621) * RG["se"]
    RG.to_csv(C.RESULTS / "revision_regimes_ci.csv")
    OUT["regime_ci"] = {k: {"delta": float(RG.loc[k, "delta"]), "lo": float(RG.loc[k, "lo95"]), "hi": float(RG.loc[k, "hi95"]),
                            "mde80": float(RG.loc[k, "mde80"])} for k in RG.index}
    # average of the four momentum survivors
    post = (base.index >= pd.Period(C.REGIME_BREAK, "M")).astype(float)
    y = base[MOMS].mean(axis=1)
    d = pd.DataFrame({"y": y, "post": post}, index=base.index).dropna()
    res = sm.OLS(d["y"], sm.add_constant(d["post"])).fit(cov_type="HAC", cov_kwds={"maxlags": I.nw_lag(len(d)), "use_correction": False})
    se = float(res.bse["post"])
    OUT["regime_mom4"] = {"pre": float(d.loc[d["post"] == 0, "y"].mean()), "post": float(d.loc[d["post"] == 1, "y"].mean()),
                          "delta": float(res.params["post"]), "t": float(res.tvalues["post"]),
                          "lo": float(res.params["post"] - 1.96 * se), "hi": float(res.params["post"] + 1.96 * se)}

    # ---------------- 6. redundancy among the survivors -------------------------------------------
    S = base[SURV].dropna()
    cm = S.corr()
    OUT["surv_corr"] = {a: {b: float(cm.loc[a, b]) for b in SURV} for a in SURV}
    mom_c = cm.loc[MOMS, MOMS].values[np.triu_indices(4, 1)]
    OUT["surv_corr_mom_min"], OUT["surv_corr_mom_max"] = float(mom_c.min()), float(mom_c.max())
    ev = np.sort(np.linalg.eigvalsh(cm.loc[MOMS, MOMS].values))[::-1]
    OUT["surv_mom_eig1_share"] = float(ev[0] / ev.sum())
    OUT["surv_neff_liji"] = float(I.neff(cm.values)["li_ji"])
    OUT["surv_neff_liji_mom"] = float(I.neff(cm.loc[MOMS, MOMS].values)["li_ji"])
    span = {}
    for k in SURV:
        others = [m for m in MOMS if m != k]
        for name, X in (("others", S[others]), ("high52", S[["HIGH52"]] if k != "HIGH52" else None)):
            if X is None:
                continue
            r = sm.OLS(S[k], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": I.nw_lag(len(S)), "use_correction": False})
            span.setdefault(k, {})[name] = {"alpha": float(r.params["const"]), "t": float(r.tvalues["const"]), "r2": float(r.rsquared)}
    OUT["spanning"] = span
    span2 = {}
    for k in ("MOM6_1", "MOM12_1"):
        r = sm.OLS(S[k], sm.add_constant(S[["HIGH52", "RESMOM"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": I.nw_lag(len(S)), "use_correction": False})
        span2[k] = {"alpha": float(r.params["const"]), "t": float(r.tvalues["const"]), "r2": float(r.rsquared)}
    OUT["spanning_h52_resmom"] = span2
    # one composite momentum factor: equal-weighted average of the four momentum survivors
    comp = base[MOMS].mean(axis=1)
    OUT["mom_composite"] = {"mean": tt(comp)[0], "t": tt(comp)[1],
                            "sr_ann": float(comp.mean() / comp.std() * np.sqrt(12))}

    (C.RESULTS / "summary_revision.json").write_text(json.dumps(OUT, indent=1, default=float))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
