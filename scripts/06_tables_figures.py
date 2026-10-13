"""All paper tables (LaTeX, booktabs) and figures (PDF) generated from results/ — no numbers are typed by
hand. Also writes paper/numbers.tex with \\newcommand macros for every number quoted in the text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402

R = C.RESULTS
TAB = C.ROOT / "paper" / "tables"
FIG = C.ROOT / "paper" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, ORANGE, GREY, GREEN, PURPLE, RED = "#1f4e79", "#d55e00", "#8c8c8c", "#009e73", "#6a3d9a", "#b2182b"
CLUSTER_COLORS = {"Momentum": BLUE, "Reversal": ORANGE, "Seasonality": PURPLE, "Low risk": GREEN,
                  "Skewness/downside": "#cc79a7", "Size/liquidity": "#e69f00", "Payout": GREY}
LABEL = {"MOM12_1": "MOM12\\_1", "MOM6_1": "MOM6\\_1", "MOM12_7": "MOM12\\_7"}
lab = lambda k: LABEL.get(k, k)  # noqa: E731  (LaTeX tables)
flab = lambda k: k  # noqa: E731  (figures)
ORDER = CH.CHAR_IDS
NUM: dict[str, str] = {}


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace("-", "$-$")


def num(name: str, value, d=2):
    if isinstance(value, (int, np.integer)):
        NUM[name] = str(int(value))
    else:
        NUM[name] = f"{value:.{d}f}"


def load_ladder(v):
    return pd.read_csv(R / f"ladder_{v}.csv", index_col=0).reindex(ORDER)


def load_factors(v):
    F = pd.read_parquet(R / "factors" / f"{v}.parquet")
    F.index = pd.PeriodIndex(F.index, freq="M")
    return F[ORDER]


# ==================================================================================
def table_sample():
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    panel = panel[(panel["month"] + 1 >= pd.Period(C.SAMPLE_START, "M"))]
    mm = pd.read_parquet(C.INTERIM / "market_monthly.parquet")
    mm.index = pd.PeriodIndex(mm.index, freq="M")
    rows = []
    for y, g in panel.groupby((panel["month"] + 1).dt.year):
        months = g["month"].unique()
        nall = g.groupby("month")["UALL"].sum().mean()
        u = g[g["U100"]]
        floor = u.groupby("month")["med12_value"].min().median() / 1e6
        cap_med = u.groupby("month")["cap"].median().median() / 1e9
        cov = u.groupby("month")["cap"].apply(lambda s: s.notna().mean()).mean()
        payers = u.groupby("month")["DY"].apply(lambda s: (s > 0).sum()).mean()
        hm = pd.PeriodIndex([m + 1 for m in months], freq="M")
        mret = (1 + mm["mkt_ret"].reindex(hm)).prod() - 1
        rfy = (1 + mm["rf"].reindex(hm)).prod() - 1
        rows.append((y, len(months), nall, floor, cap_med, 100 * cov, payers, 100 * mret, 100 * rfy))
    df = pd.DataFrame(rows, columns=["year", "months", "nall", "floor", "capmed", "capcov", "payers", "mkt", "rf"])
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Year & Months & $N_{\text{ALL}}$ & U100 floor & Median cap & Cap coverage & Payers & MCFTRR & $r_f$ \\",
             r" & & & (RUB m/month) & (RUB bn) & (\%) & in U100 & (\%) & (\%) \\", r"\midrule"]
    for r in df.itertuples():
        lines.append(f"{r.year} & {r.months} & {r.nall:.0f} & {r.floor:.1f} & {r.capmed:.1f} & {r.capcov:.0f} & {r.payers:.0f} & {fmt(r.mkt, 1)} & {r.rf:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_sample.tex").write_text("\n".join(lines))
    num("NallMean", float(df["nall"].mean()), 0)
    num("FloorFirst", float(df["floor"].iloc[0]), 0)
    num("FloorLast", float(df["floor"].iloc[-1]), 0)
    num("CapCovFirst", float(df["capcov"].iloc[0]), 0)
    num("PayersMean", float(df["payers"].mean()), 0)
    return df


def table_definitions():
    src = {"MOM12_1": r"\citet{jegadeesh1993returns}", "MOM6_1": r"\citet{jegadeesh1993returns}",
           "MOM12_7": r"\citet{novymarx2012momentum}", "RESMOM": r"\citet{blitz2011residual}",
           "HIGH52": r"\citet{george2004week}", "STR": r"\citet{jegadeesh1990evidence}",
           "LTR": r"\citet{debondt1985overreact}", "SEAS": r"\citet{heston2008seasonality}",
           "BETA": r"\citet{frazzini2014betting}", "IVOL": r"\citet{ang2006cross}", "TVOL": r"\citet{ang2006cross}",
           "MAX": r"\citet{bali2011maxing}", "DBETA": r"\citet{ang2006downside}",
           "SKEW": r"\citet{amaya2015realized}", "ISKEW": r"\citet{boyer2010expected}",
           "COSKEW": r"\citet{harvey2000conditional}", "SIZE": r"\citet{banz1981relationship}",
           "AMIHUD": r"\citet{amihud2002illiquidity}", "DVOL": r"\citet{brennan1998alternative}",
           "TURN": r"\citet{datar1998liquidity}", "ZERO": r"\citet{lesmond1999new}; \citet{bekaert2007liquidity}",
           "SPREAD": r"\citet{corwin2012simple}; \citet{amihud1986asset}", "CVVAL": r"\citet{chordia2001trading}",
           "ABVOL": r"\citet{gervais2001high}", "DY": r"\citet{litzenberger1979effect}"}
    d = {"MOM12_1": r"$\prod_{j=1}^{11}(1+R_{t-j})-1$",
         "MOM6_1": r"$\prod_{j=1}^{5}(1+R_{t-j})-1$",
         "MOM12_7": r"$\prod_{j=6}^{11}(1+R_{t-j})-1$",
         "RESMOM": r"CAPM residuals $t{-}11..t{-}1$ / their s.d.; betas from $t{-}35..t{-}1$",
         "HIGH52": r"$P_t / \max P$ over 252 sessions",
         "STR": r"$R_t$", "LTR": r"$\prod_{j=12}^{35}(1+R_{t-j})-1$",
         "SEAS": r"mean of $R_{t+1-12j}$, $j=1..5$",
         "BETA": r"Dimson beta ($m_d$, $m_{d-1}$), 252 sessions",
         "IVOL": r"s.d.\ of CAPM residuals, month $t$", "TVOL": r"s.d.\ of $r_d$, month $t$",
         "MAX": r"$\max r_d$, month $t$", "DBETA": r"beta on days with $m_d<\bar m$, 252 sessions",
         "SKEW": r"skewness of $r_d$, 252 sessions", "ISKEW": r"skewness of CAPM residuals, 252 sessions",
         "COSKEW": r"$E[\varepsilon_i\varepsilon_m^2]/(\sqrt{E\varepsilon_i^2}E\varepsilon_m^2)$, 252 sessions",
         "SIZE": r"$\log$ market cap$_t$", "AMIHUD": r"mean $|r_d|/\text{value}_d$, 252 sessions",
         "DVOL": r"$\log$ mean daily value, 252 sessions", "TURN": r"$\sum$ value$_{t-11..t}$ / mean cap",
         "ZERO": r"share of no-trade or zero-return sessions, 252 sessions",
         "SPREAD": r"Corwin--Schultz spread, month $t$", "CVVAL": r"s.d./mean of monthly value, $t{-}11..t$",
         "ABVOL": r"$\log$(daily value in $t$ / daily value $t{-}12..t{-}1$)",
         "DY": r"12-month gross dividends / $P_t$ (payers only)"}
    lines = [r"\begin{tabular}{llp{6.3cm}cl}", r"\toprule",
             r"ID & Cluster & Definition (at the last session of month $t$) & Sign & Source \\", r"\midrule"]
    prev = None
    for k, cl, s in CH.CHARS:
        if prev is not None and cl != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = cl
        sg = "$+$" if s > 0 else "$-$"
        lines.append(f"{lab(k)} & {cl} & {d[k]} & {sg} & {src[k]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_definitions.tex").write_text("\n".join(lines))


def table_baseline():
    L = load_ladder("baseline")
    Cp = load_ladder("capm")
    rungs = ["R1_t196", "R2_hlz3", "R3_holm", "R4_bh", "R5_by", "R6_rw", "R7_dsr"]
    lines = [r"\begin{tabular}{lrrrrrrr ccccccc r}", r"\toprule",
             r" & \multicolumn{7}{c}{Long--short return (predicted direction)} & \multicolumn{7}{c}{Discovery under rung} & \\",
             r"\cmidrule(lr){2-8}\cmidrule(lr){9-15}",
             r"Factor & Mean & $t$ & SR & $\alpha$ & $t(\alpha)$ & $\beta$ & Skew & R1 & R2 & R3 & R4 & R5 & R6 & R7 & $p_{\text{RW}}$ \\",
             r"\midrule"]
    prev = None
    for k in ORDER:
        cl = CH.CLUSTER[k]
        if prev is not None and cl != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = cl
        r = L.loc[k]
        c = Cp.loc[k]
        marks = []
        for rg in rungs:
            if bool(r[rg]):
                marks.append(r"\checkmark")
            elif rg == "R6_rw" and bool(r["wrong_sign_rw"]):
                marks.append(r"$\times$")
            elif rg == "R1_t196" and bool(r["wrong_sign_t196"]):
                marks.append(r"$\times$")
            else:
                marks.append("")
        lines.append(f"{lab(k)} & {fmt(100 * r['mean'])} & {fmt(r['t'])} & {fmt(r['sr_ann'])} & {fmt(100 * c['alpha'])} & "
                     f"{fmt(c['t_alpha'])} & {fmt(c['beta_mkt'])} & {fmt(r['skew'])} & " + " & ".join(marks)
                     + f" & {fmt(r['p_rw'], 3)} \\\\")
    cnt = {rg: int(L[rg].sum()) for rg in rungs}
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{Number of discoveries (of 25)} & " + " & ".join(str(cnt[rg]) for rg in rungs) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_baseline.tex").write_text("\n".join(lines))
    for rg in rungs:
        num("Count" + rg.split("_")[0], cnt[rg])
    for k in ORDER:
        kk = k.replace("_", "")
        num(f"Mean{kk}", 100 * L.loc[k, "mean"])
        num(f"T{kk}", L.loc[k, "t"])
        num(f"SR{kk}", L.loc[k, "sr_ann"])
        num(f"Prw{kk}", L.loc[k, "p_rw"], 3)
        num(f"Alpha{kk}", 100 * Cp.loc[k, "alpha"])
        num(f"Talpha{kk}", Cp.loc[k, "t_alpha"])
        num(f"Dsr{kk}", L.loc[k, "dsr"])
    num("TabsSTR", abs(L.loc["STR", "t"]))
    num("MeanAbsSTR", abs(100 * L.loc["STR", "mean"]))


def table_ladder_counts():
    S = json.loads((R / "summary.json").read_text())
    specs = [("baseline", "Baseline (U100, EW, quintiles)"), ("baseline_block3", r"\quad RW mean block = 3"),
             ("baseline_block12", r"\quad RW mean block = 12"), ("vw", "Capped value weights"),
             ("u50", "U50 (50 most liquid)"), ("uall", "UALL (all active shares)"), ("terciles", "Terciles"),
             ("lag1", "One-session implementation lag"), ("u100ord", "Preferred duplicates dropped"),
             ("priceonly", "Price-only holding returns"), ("extrans", "Excluding 2022-02, 2022-03"),
             ("pit", "Point-in-time SIZE/TURN"), ("capm", "CAPM alpha (market-hedged)"),
             ("costs", "Net of trading costs"), ("regime_pre", "Pre-2022 (2012-01--2022-02)"),
             ("regime_post", "Post-2022 (2022-03--2026-09)")]
    rungs = ["R1_t196", "R2_hlz3", "R3_holm", "R4_bh", "R5_by", "R6_rw", "R7_dsr"]
    lines = [r"\begin{tabular}{l ccccccc cc}", r"\toprule",
             r"Specification & R1 & R2 & R3 & R4 & R5 & R6 & R7 & Wrong sign & RW crit. \\",
             r" & $t{>}1.96$ & $t{>}3$ & Holm & BH & BY & RW & DSR & (RW) & (step 1) \\", r"\midrule"]
    for key, name in specs:
        info = S[key]
        c = info["counts"]
        vals = [str(c[rg]) for rg in rungs]
        ws = str(c.get("wrong_sign_rw", ""))
        crit = fmt(info.get("rw_crit_first", np.nan))
        lines.append(f"{name} & " + " & ".join(vals) + f" & {ws} & {crit} \\\\")
        if key == "baseline_block12" or key == "pit" or key == "costs":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_ladder_counts.tex").write_text("\n".join(lines))
    num("RWcrit", S["baseline"]["rw_crit_first"])
    num("RWcritB3", S["baseline_block3"]["rw_crit_first"])
    num("RWcritB12", S["baseline_block12"]["rw_crit_first"])
    num("GlobalNullP", S["baseline"]["global_null_p"], 4)
    num("NeffLiJi", S["neff"]["li_ji"], 0)
    num("NeffNyholt", S["neff"]["nyholt"], 1)
    num("EigOne", S["neff"]["eigen_top5"][0], 2)
    num("EigShare", 100 * S["neff"]["eigen_top5"][0] / 25, 0)
    num("MeanAbsCorr", S["neff"]["corr_mean_abs_offdiag"], 2)
    num("PBO", S["pbo"]["pbo"], 3)
    num("PBOT", S["pbo"]["T_common"])
    num("SRzeroN", np.sqrt(12) * S["dsr"]["sr0_N25"])
    num("SRzeroNeff", np.sqrt(12) * S["dsr"]["sr0_Neff"])
    num("SRzeroAll", np.sqrt(12) * S["dsr"]["sr0_N250"])
    num("EBmuzero", 100 * S["eb"]["mu0"])
    num("EBtauc", 100 * S["eb"]["tau_c"])
    num("EBtauw", 100 * S["eb"]["tau_w"])
    num("EBnrep", S["eb"]["n_replicate_pos"])
    num("EBrate", 100 * S["eb"]["replication_rate"], 0)
    num("RegZooDelta", 100 * S["regimes"]["zoo_delta"])
    num("RegZooT", S["regimes"]["zoo_t_delta"])
    num("RegBHsig", S["regimes"]["n_bh_sig"])
    num("RWcritPost", S["regime_post"]["rw_crit_first"])
    num("RWcritPre", S["regime_pre"]["rw_crit_first"])
    for key in ("vw", "u50", "uall", "terciles", "lag1", "u100ord", "priceonly", "extrans", "pit", "capm", "costs",
                "regime_pre", "regime_post"):
        num("RW" + key.replace("_", "").replace("regime", "reg"), S[key]["counts"]["R6_rw"])
    bonf = stats.norm.ppf(1 - 0.025 / 25)
    num("BonfCrit", bonf)
    E = json.loads((R / "summary_extensions.json").read_text())
    num("SRzeroBoot", np.sqrt(12) * E["dsr_boot_sr0_monthly"])
    num("DSRbootPass", E["dsr_boot_pass"])
    num("EBncPos", E["eb_nocluster"]["n_pos"])


def table_regimes():
    G = pd.read_csv(R / "regimes.csv", index_col=0)
    Lpre, Lpost = load_ladder("pre"), load_ladder("post")
    lines = [r"\begin{tabular}{l rr rr rrrc}", r"\toprule",
             r" & \multicolumn{2}{c}{2012-01--2022-02} & \multicolumn{2}{c}{2022-03--2026-09} & \multicolumn{4}{c}{Difference} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}",
             r"Factor & Mean & $t$ & Mean & $t$ & $\Delta$ & $t(\Delta)$ & $p_{\text{BH}}$ & 95\% CI \\", r"\midrule"]
    prev = None
    for k in ORDER + ["ZOO_EW"]:
        if k != "ZOO_EW":
            cl = CH.CLUSTER[k]
            if prev is not None and cl != prev:
                lines.append(r"\addlinespace[2pt]")
            prev = cl
        r = G.loc[k]
        name = lab(k) if k != "ZOO_EW" else r"\midrule Average of 25"
        pb = fmt(r["p_delta_bh"], 2) if k != "ZOO_EW" else "--"
        se = r["delta"] / r["t_delta"]
        ci = f"[{fmt(100 * (r['delta'] - 1.96 * se))}, {fmt(100 * (r['delta'] + 1.96 * se))}]"
        lines.append(f"{name} & {fmt(100 * r['mean_pre'])} & {fmt(r['t_pre'])} & {fmt(100 * r['mean_post'])} & {fmt(r['t_post'])} & "
                     f"{fmt(100 * r['delta'])} & {fmt(r['t_delta'])} & {pb} & {ci} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_regimes.tex").write_text("\n".join(lines))
    for k in ["DVOL", "ZERO", "AMIHUD", "BETA", "MOM6_1", "HIGH52", "STR", "SEAS"]:
        kk = k.replace("_", "")
        num(f"RegPre{kk}", 100 * G.loc[k, "mean_pre"])
        num(f"RegPost{kk}", 100 * G.loc[k, "mean_post"])
        num(f"RegTd{kk}", G.loc[k, "t_delta"])
        num(f"RegTpost{kk}", G.loc[k, "t_post"])
        num(f"RegTpre{kk}", G.loc[k, "t_pre"])
    num("RegZooPre", 100 * G.loc["ZOO_EW", "mean_pre"])
    num("RegZooPost", 100 * G.loc["ZOO_EW", "mean_post"])
    num("RegZooTpre", G.loc["ZOO_EW", "t_pre"])
    num("RegZooTpost", G.loc["ZOO_EW", "t_post"])
    num("RegMinPdvol", G.loc[["DVOL", "ZERO"], "p_delta_bh"].min())
    num("RegRawPdvol", G.loc["DVOL", "p_delta"], 3)


def table_costs():
    Co = pd.read_csv(R / "costs.csv", index_col=0).reindex(ORDER)
    lines = [r"\begin{tabular}{l rrr rr rr r c}", r"\toprule",
             r" & \multicolumn{3}{c}{Gross} & \multicolumn{2}{c}{Net (1$\times$ cost)} & \multicolumn{2}{c}{Net (2$\times$ cost)} & Break-even & RW net \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Factor & Mean & Turnover & Cost & Mean & $t$ & Mean & $t$ & (bp, one-way) & \\", r"\midrule"]
    prev = None
    for k in ORDER:
        cl = CH.CLUSTER[k]
        if prev is not None and cl != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = cl
        r = Co.loc[k]
        be = r["breakeven_oneway_bp"]
        bes = fmt(be, 0) if be > 0 else "--"
        mark = r"\checkmark" if bool(r["net_R6_rw"]) else ""
        lines.append(f"{lab(k)} & {fmt(100 * r['gross_mean'])} & {fmt(r['turnover_2sided'])} & {fmt(100 * r['avg_cost'])} & "
                     f"{fmt(100 * r['net_mean'])} & {fmt(r['net_t'])} & {fmt(100 * r['net2x_mean'])} & {fmt(r['net2x_t'])} & {bes} & {mark} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_costs.tex").write_text("\n".join(lines))
    for k in ["MOM12_1", "MOM6_1", "RESMOM", "HIGH52", "ABVOL", "MOM12_7"]:
        kk = k.replace("_", "")
        num(f"NetMean{kk}", 100 * Co.loc[k, "net_mean"])
        num(f"BE{kk}", Co.loc[k, "breakeven_oneway_bp"], 0)
        num(f"NetT{kk}", Co.loc[k, "net_t"])
        num(f"NetTwo{kk}", Co.loc[k, "net2x_t"])
        num(f"TO{kk}", Co.loc[k, "turnover_2sided"])
    num("NetRWcount", int(Co["net_R6_rw"].sum()))
    num("CashNamesMax", int(Co["cash_names_total"].max()))


def table_fm():
    U = pd.read_csv(R / "fm_univariate.csv", index_col=0).reindex(ORDER)
    M = pd.read_csv(R / "fm_multivariate.csv", index_col=0).reindex(ORDER)
    X = pd.read_csv(R / "exploratory_fm.csv", index_col=0)
    A = X[X["spec"] == "A_one_per_cluster"]
    B = X[X["spec"] == "B_survivors_plus_STR"]
    lines = [r"\begin{tabular}{l rr rr rr rr}", r"\toprule",
             r" & \multicolumn{2}{c}{Univariate} & \multicolumn{2}{c}{All 25 jointly} & \multicolumn{2}{c}{One per cluster$^\ast$} & \multicolumn{2}{c}{Survivors + STR$^\ast$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Characteristic & Slope & $t$ & Slope & $t$ & Slope & $t$ & Slope & $t$ \\", r"\midrule"]
    prev = None
    for k in ORDER:
        cl = CH.CLUSTER[k]
        if prev is not None and cl != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = cl
        a = (fmt(100 * A.loc[k, "mean"]), fmt(A.loc[k, "t"])) if k in A.index else ("", "")
        b = (fmt(100 * B.loc[k, "mean"]), fmt(B.loc[k, "t"])) if k in B.index else ("", "")
        lines.append(f"{lab(k)} & {fmt(100 * U.loc[k, 'mean'])} & {fmt(U.loc[k, 't'])} & {fmt(100 * M.loc[k, 'mean'])} & {fmt(M.loc[k, 't'])} & "
                     f"{a[0]} & {a[1]} & {b[0]} & {b[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_fm.tex").write_text("\n".join(lines))
    num("FMmaxTmulti", float(M["t"].abs().max()))
    num("FMtMOMA", float(A.loc["MOM12_1", "t"]))
    num("FMtSTRA", float(A.loc["STR", "t"]))
    num("FMtMOMB", float(B.loc["MOM12_1", "t"]))
    num("FMtHIGHB", float(B.loc["HIGH52", "t"]))
    num("FMtSTRB", float(B.loc["STR", "t"]))
    num("FMtABVOLB", float(B.loc["ABVOL", "t"]))
    num("FMtHIGHuni", float(U.loc["HIGH52", "t"]))


def table_power():
    P = pd.read_csv(R / "power.csv")
    piv = P.pivot(index="sr_ann", columns="rung", values="power_mean")
    lines = [r"\begin{tabular}{c cccccc}", r"\toprule",
             r"True annual SR & R1 $t{>}1.96$ & R2 $t{>}3$ & R3 Holm & R4 BH & R5 BY & R6 RW \\", r"\midrule"]
    for sr, r in piv.iterrows():
        lines.append(f"{sr:.2f} & " + " & ".join(f"{v:.2f}" for v in r[["R1", "R2", "R3", "R4", "R5", "R6"]].values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_power.tex").write_text("\n".join(lines))
    for sr in [0.5, 0.75, 1.0, 1.25]:
        s = str(sr).replace(".", "p")
        num(f"PowRWsr{s}", piv.loc[sr, "R6"])
        num(f"PowTsr{s}", piv.loc[sr, "R1"])


def table_robust_t():
    vs = [("baseline", "Base"), ("vw", "VW"), ("u50", "U50"), ("uall", "UALL"), ("terciles", "Terc."),
          ("lag1", "Lag"), ("u100ord", "Ord."), ("priceonly", "Price"), ("extrans", "Ex-22"), ("pit", "PIT"),
          ("capm", r"Hedged")]
    T = {}
    Rw = {}
    for v, _ in vs:
        L = load_ladder(v)
        T[v] = L["t"]
        Rw[v] = L["R6_rw"] | L["wrong_sign_rw"]
    lines = [r"\begin{tabular}{l" + "r" * len(vs) + "}", r"\toprule",
             "Factor & " + " & ".join(n for _, n in vs) + r" \\", r"\midrule"]
    prev = None
    for k in ORDER:
        cl = CH.CLUSTER[k]
        if prev is not None and cl != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = cl
        cells = []
        for v, _ in vs:
            t = T[v][k]
            s = fmt(t)
            if bool(Rw[v][k]):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{lab(k)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_robust_t.tex").write_text("\n".join(lines))


def table_eb():
    E = pd.read_csv(R / "eb_baseline.csv", index_col=0).reindex(ORDER)
    N = pd.read_csv(R / "exploratory_eb_nocluster.csv", index_col=0).reindex(ORDER)
    lines = [r"\begin{tabular}{ll r rrr r rr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Clustered prior (pre-specified)} & \multicolumn{2}{c}{No clusters$^\ast$} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
             r"Factor & Cluster & OLS mean & Post. mean & 2.5\% & 97.5\% & $P(\mu>0)$ & Post. mean & $P(\mu>0)$ \\", r"\midrule"]
    prev = None
    for k in ORDER:
        cl = CH.CLUSTER[k]
        if prev is not None and cl != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = cl
        e, n = E.loc[k], N.loc[k]
        lines.append(f"{lab(k)} & {cl} & {fmt(100 * e['mean'])} & {fmt(100 * e['post_mean'])} & {fmt(100 * e['lo95'])} & {fmt(100 * e['hi95'])} & "
                     f"{e['p_pos']:.2f} & {fmt(100 * n['post_mean'])} & {n['p_pos']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_eb.tex").write_text("\n".join(lines))


def table_str_turnover():
    D = pd.read_csv(R / "exploratory_str_turnover_diff.csv", index_col=0)
    M = json.loads((R / "summary_mechanism.json").read_text())
    names = {"low_turn": "Low turnover", "mid_turn": "Medium turnover", "high_turn": "High turnover",
             "high_minus_low": "High $-$ low"}
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r" & Winners $-$ losers of month $t$ & $t$ \\", r"\midrule",
             r"\multicolumn{3}{l}{\textit{A. All active shares (UALL), turnover terciles}} \\"]
    for k, n in names.items():
        lines.append(f"\\quad {n} & {fmt(100 * D.loc[k, 'mean'])} & {fmt(D.loc[k, 't'])} \\\\")
    lines.append(r"\addlinespace[3pt]")
    lines.append(r"\multicolumn{3}{l}{\textit{B. Liquid universe (U100), turnover halves}} \\")
    for k, n in (("u100_low_turn", "Low turnover"), ("u100_high_turn", "High turnover"), ("u100_high_minus_low", "High $-$ low")):
        lines.append(f"\\quad {n} & {fmt(100 * M[k + '_mean'])} & {fmt(M[k + '_t'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_str_turnover.tex").write_text("\n".join(lines))
    num("STRlowM", 100 * D.loc["low_turn", "mean"])
    num("STRlowT", D.loc["low_turn", "t"])
    num("STRhighM", 100 * D.loc["high_turn", "mean"])
    num("STRhighT", D.loc["high_turn", "t"])
    num("STRdiffM", 100 * D.loc["high_minus_low", "mean"])
    num("STRdiffT", D.loc["high_minus_low", "t"])


# ==================================================================================
def fig_sample(df_years):
    U = pd.read_parquet(C.PROCESSED / "universes.parquet")
    U["month"] = pd.PeriodIndex(U["month"], freq="M")
    U = U[(U["month"] >= pd.Period(C.FIRST_FORMATION, "M")) & (U["month"] <= pd.Period(C.SAMPLE_END, "M") - 1)]
    n = U.groupby("month")["UALL"].sum()
    floor = U[U["U100"]].groupby("month")["med12_value"].min() / 1e6
    f50 = U[U["U50"]].groupby("month")["med12_value"].min() / 1e6
    x = n.index.to_timestamp()
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
    ax[0].plot(x, n.values, color=BLUE, lw=1.1)
    ax[0].set_ylabel("Active shares (UALL)")
    ax[0].set_ylim(0, max(n.values) * 1.1)
    ax[0].axvline(pd.Timestamp("2022-02-25"), color=GREY, lw=0.6, ls="--")
    ax[0].set_title("(a) Cross-section")
    ax[1].semilogy(x, floor.values, color=BLUE, lw=1.1, label="100th share (U100)")
    ax[1].semilogy(x, f50.reindex(n.index).values, color=ORANGE, lw=1.1, label="50th share (U50)")
    ax[1].set_ylabel("Median monthly value, RUB m")
    ax[1].legend(frameon=False, loc="upper left")
    ax[1].axvline(pd.Timestamp("2022-02-25"), color=GREY, lw=0.6, ls="--")
    ax[1].set_title("(b) Liquidity floor of the universes")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sample.pdf")
    plt.close(fig)


def fig_ladder():
    L = load_ladder("baseline")
    S = json.loads((R / "summary.json").read_text())
    crit = S["baseline"]["rw_crit_first"]
    d = L.sort_values("t")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(d))
    cols = [CLUSTER_COLORS[CH.CLUSTER[k]] for k in d.index]
    ax.hlines(y, 0, d["t"], color=cols, lw=2.2)
    ax.scatter(d["t"], y, color=cols, s=14, zorder=3)
    thr_handles = []
    for xv, ls, txt in [(1.96, ":", "$|t| = 1.96$"), (3.0, "--", "Harvey–Liu–Zhu $3.00$"),
                        (stats.norm.ppf(1 - 0.025 / 25), "-.", "Bonferroni (normal) $%.2f$" % stats.norm.ppf(1 - 0.025 / 25)),
                        (crit, "-", "Romano–Wolf (bootstrap) $%.2f$" % crit)]:
        ax.axvline(xv, color="k", lw=0.7, ls=ls)
        ax.axvline(-xv, color="k", lw=0.7, ls=ls)
        thr_handles.append(plt.Line2D([0], [0], color="k", lw=0.8, ls=ls, label=txt))
    ax.axvline(0, color=GREY, lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([flab(k) for k in d.index], fontsize=7)
    ax.set_xlabel("Newey–West $t$-statistic of the long–short return (positive = predicted direction)")
    handles = [plt.Line2D([0], [0], color=c, lw=3) for c in CLUSTER_COLORS.values()]
    leg1 = ax.legend(handles, list(CLUSTER_COLORS.keys()), frameon=False, loc="lower right", fontsize=7)
    ax.add_artist(leg1)
    lg = ax.legend(handles=thr_handles, frameon=True, framealpha=1.0, edgecolor="none", facecolor="white", loc="upper left", fontsize=7, title="Two-sided thresholds", title_fontsize=7)
    lg.set_zorder(5)
    ax.set_ylim(-0.7, len(d) - 0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ladder.pdf")
    plt.close(fig)


def fig_boot():
    b = np.load(R / "boot_baseline.npz")
    mx = b["maxabs"]
    S = json.loads((R / "summary.json").read_text())
    crit = S["baseline"]["rw_crit_first"]
    zc = np.load(R / "mech_corr_normal_max.npy")
    iid = np.load(R / "mech_iid_nw_corr_max.npy")
    bonf = float(stats.norm.ppf(1 - 0.025 / 25))
    Mj = json.loads((R / "summary_mechanism.json").read_text())
    q_b, q_i, q_z = crit, Mj["q_iidnw_corr_L4"], Mj["q_normal_corr"]  # quantiles from the full simulations
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bins = np.linspace(0, 7, 106)
    ax.hist(mx, bins=bins, density=True, color=BLUE, alpha=0.5,
            label="Bootstrap of the 25 MOEX factors (95%% quantile %.2f)" % q_b)
    ax.hist(iid, bins=bins, density=True, histtype="step", color=GREEN, lw=1.2,
            label="Newey–West $t$, i.i.d. normal returns, $T=177$, factor correlations (%.2f)" % q_i)
    ax.hist(zc, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.2,
            label="Standard normal statistics with factor correlations (%.2f)" % q_z)
    ax.axvline(q_b, color=BLUE, lw=1.0)
    ax.axvline(q_i, color=GREEN, lw=1.0, ls="-.")
    ax.axvline(q_z, color=ORANGE, lw=1.0, ls="--")
    ax.axvline(bonf, color="k", lw=0.8, ls=":", label="Bonferroni bound for 25 normal tests (%.2f)" % bonf)
    ax.set_xlabel("Maximum of 25 absolute $t$-statistics under the null (vertical lines: 95% quantiles)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax.set_xlim(0, 7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_boot.pdf")
    plt.close(fig)
    num("BootMaxMedian", float(np.median(mx)))


def fig_cumret():
    F = load_factors("baseline")
    L = load_ladder("baseline")
    surv = list(L.index[L["R6_rw"]])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    pal = [BLUE, ORANGE, GREEN, PURPLE, "#e69f00", "#cc79a7"]
    for i, k in enumerate(surv):
        s = F[k].fillna(0.0)
        ax.plot(s.index.to_timestamp(), np.log1p(s).cumsum(), color=pal[i % len(pal)], lw=1.1, label=flab(k))
    s = -F["STR"].fillna(0.0)
    ax.plot(s.index.to_timestamp(), np.log1p(s).cumsum(), color=GREY, lw=1.1, ls="--", label=r"$-$STR (continuation)")
    ax.axvline(pd.Timestamp("2022-02-25"), color="k", lw=0.6, ls=":")
    ax.text(pd.Timestamp("2022-04-01"), ax.get_ylim()[1] * 0.95, "foreign-investor lockout", fontsize=7, va="top")
    ax.set_ylabel("Cumulative log return")
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_cumret.pdf")
    plt.close(fig)


def fig_regimes():
    G = pd.read_csv(R / "regimes.csv", index_col=0).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    texts = []
    for k in ORDER:
        c = CLUSTER_COLORS[CH.CLUSTER[k]]
        ax.scatter(G.loc[k, "t_pre"], G.loc[k, "t_post"], color=c, s=16, zorder=3)
        texts.append(ax.text(G.loc[k, "t_pre"], G.loc[k, "t_post"], flab(k), fontsize=5.5))
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.3))
    except Exception:
        pass
    lim = [-5.5, 5.5]
    ax.plot(lim, lim, color=GREY, lw=0.6)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(0, color=GREY, lw=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("$t$, 2012-01 – 2022-02")
    ax.set_ylabel("$t$, 2022-03 – 2026-09")
    fig.tight_layout()
    fig.savefig(FIG / "fig_regimes.pdf")
    plt.close(fig)


def fig_power():
    P = pd.read_csv(R / "power.csv")
    piv = P.pivot(index="sr_ann", columns="rung", values="power_mean")
    names = {"R1": r"R1 $t>1.96$", "R2": r"R2 $t>3$", "R3": "R3 Holm", "R4": "R4 BH", "R5": "R5 BY", "R6": "R6 Romano–Wolf"}
    sty = {"R1": (GREY, "-"), "R2": (ORANGE, "--"), "R3": (GREEN, ":"), "R4": (PURPLE, "-."), "R5": ("#cc79a7", ":"), "R6": (BLUE, "-")}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, n in names.items():
        c, ls = sty[k]
        ax.plot(piv.index, piv[k], color=c, ls=ls, lw=1.3 if k == "R6" else 1.0, marker="o", ms=2.5, label=n)
    ax.axhline(0.8, color="k", lw=0.5, ls=":")
    ax.set_xlabel("True annualised Sharpe ratio of one factor")
    ax.set_ylabel("Probability of discovery")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_power.pdf")
    plt.close(fig)


def fig_eb():
    E = pd.read_csv(R / "eb_baseline.csv", index_col=0).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(E))
    cols = [CLUSTER_COLORS[CH.CLUSTER[k]] for k in E.index]
    ax.scatter(x - 0.15, 100 * E["mean"], marker="o", facecolors="none", edgecolors=cols, s=18, label="OLS mean")
    for i, k in enumerate(E.index):
        ax.plot([x[i] + 0.15, x[i] + 0.15], [100 * E.loc[k, "lo95"], 100 * E.loc[k, "hi95"]], color=cols[i], lw=1.0)
    ax.scatter(x + 0.15, 100 * E["post_mean"], color=cols, s=14, label="Posterior mean, 95% interval")
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([flab(k) for k in E.index], rotation=70, fontsize=6.5)
    ax.set_ylabel("Monthly premium, %")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_eb.pdf")
    plt.close(fig)


def fig_corr():
    F = load_factors("baseline")
    Cm = F.corr()
    S = json.loads((R / "summary.json").read_text())
    ev = np.array(S["neff"]["eigen_top5"])
    lam = np.sort(np.linalg.eigvalsh(Cm.values))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    im = ax[0].imshow(Cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(Cm)))
    ax[0].set_yticks(range(len(Cm)))
    ax[0].set_xticklabels([flab(k) for k in Cm.index], rotation=90, fontsize=5)
    ax[0].set_yticklabels([flab(k) for k in Cm.index], fontsize=5)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
    ax[0].set_title("(a) Correlations of long–short returns")
    ax[1].bar(np.arange(1, len(lam) + 1), lam, color=BLUE)
    ax[1].axhline(1, color=GREY, lw=0.6, ls="--")
    ax[1].set_xlabel("Eigenvalue rank")
    ax[1].set_ylabel("Eigenvalue")
    ax[1].set_title("(b) Spectrum: $N_{\\mathrm{eff}}$ = %.0f (Li–Ji)" % S["neff"]["li_ji"])
    fig.tight_layout()
    fig.savefig(FIG / "fig_corr.pdf")
    plt.close(fig)


def mechanism_macros():
    M = json.loads((R / "summary_mechanism.json").read_text())
    num("MechSDmed", M["tstar_sd_median"])
    num("MechSDmin", M["tstar_sd_min"])
    num("MechSDmax", M["tstar_sd_max"])
    num("MechKurtMed", M["tstar_kurt_median"])
    num("MechRescaled", M["crit_rescaled_unit_sd"])
    num("MechNormalScores", M["crit_normal_scores"])
    num("MechRetKurtMed", M["return_kurt_median"], 1)
    num("MechIidPthree", M["iid_nw_p309"], 4)
    num("MechIidSD", M["iid_nw_sd"], 3)
    num("MechIidCrit", M["iid_nw_max_crit"])
    num("QnormIndep", M["q_normal_indep"])
    num("QnormCorr", M["q_normal_corr"])
    num("QiidIndepL4", M["q_iidnw_indep_L4"])
    num("QiidCorrL4", M["q_iidnw_corr_L4"])
    num("QiidCorrL8", M["q_iidnw_corr_L8"])
    num("QiidCorrL12", M["q_iidnw_corr_L12"])
    num("DecompDep", M["q_normal_indep"] - M["q_normal_corr"])
    num("DecompNoise", M["q_iidnw_corr_L4"] - M["q_normal_corr"])
    num("DecompData", M["crit_baseline"] - M["q_iidnw_corr_L4"])
    num("NormMaxCrit", M["q_normal_indep"])
    num("MechSidakNeff", M["sidak_crit_neff15"])
    num("MechCritLeight", M["crit_L8"])
    num("MechCritLtwelve", M["crit_L12"])
    num("MechTratioLtwelve", 100 * (1 - M["tratio_L12"]), 0)
    num("MechPosAC", 100 * M["share_pos_autocorr_sum12"], 0)
    cr = [v["crit"] for v in M["seeds"].values()]
    num("SeedCritMin", min(cr + [M["crit_baseline"]]))
    num("SeedCritMax", max(cr))
    pab = [v["p_abvol"] for v in M["seeds"].values()]
    num("SeedPabvolMin", min(pab), 3)
    num("SeedPabvolMax", max(pab), 3)
    same = all(v["survivors"] == M["seeds"]["1"]["survivors"] for v in M["seeds"].values())
    NUM["SeedSame"] = "the same" if same else "different"
    num("FWERpar", 100 * M["fwer_sim_parametric"], 1)
    num("FWERnp", 100 * M["fwer_sim_nonparametric"], 1)
    num("FWERreps", int(M["fwer_sim_reps"]))
    num("PBOmom", 100 * M["pbo_share_momentum_best"], 0)
    num("PBOhigh", 100 * M["pbo_share_high52_best"], 0)
    num("PBOabvol", 100 * M["pbo_share_abvol_best"], 0)
    num("PBOrankMed", int(M["pbo_oos_rank_median"]))
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
    NUM["PBOrankWord"] = words.get(25 - int(M["pbo_oos_rank_median"]) + 1, str(25 - int(M["pbo_oos_rank_median"]) + 1))
    num("PBOtopfive", 100 * M["pbo_oos_top5_share"], 0)
    num("PBOsplits", int(M["pbo_splits"]))
    num("UhundTopTurn", 100 * M["u100_share_in_top_turn_tercile"], 0)
    num("UhundLowTurnM", 100 * M["u100_low_turn_mean"])
    num("UhundLowTurnT", M["u100_low_turn_t"])
    num("UhundHighTurnM", 100 * M["u100_high_turn_mean"])
    num("UhundHighTurnT", M["u100_high_turn_t"])
    num("UhundDiffM", 100 * M["u100_high_minus_low_mean"])
    num("UhundDiffT", M["u100_high_minus_low_t"])
    num("EBnineNpos", int(M["eb9_n_pos"]))
    NUM["EBnineSame"] = "the same" if sorted(M["eb9_pos"]) == sorted(M["eb_cluster_pos"]) else "a different"
    num("EBpostSDmed", 100 * M["eb_post_sd_median"])
    num("OLSseMed", 100 * M["ols_se_median"])
    num("CorrIvolTvol", M["corr_ivol_tvol"])
    num("CorrMomMom", M["corr_mom121_mom127"])
    num("MktExcess", 100 * M["mkt_excess_mean"])
    L = load_ladder("baseline")
    sv = L.index[L["R6_rw"]]
    num("SRsurvMin", float(L.loc[sv, "sr_ann"].min()))
    num("SRsurvMax", float(L.loc[sv, "sr_ann"].max()))


# ==================================================================================
# revision (exploratory analyses of scripts 05e and 05f)
# ==================================================================================

def _mt(x, d=2):
    """mean (in %) and t as two strings"""
    return fmt(100 * x[0], d), "(" + fmt(x[1], 2) + ")"


def _rub(x):
    """roubles -> millions, no decimals, thin thousands separator"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x <= 0:
        return "0"
    v = x / 1e6
    return f"{v:,.0f}".replace(",", "{,}") if v >= 1 else f"{v:.1f}"


def revision_tables():
    V = json.loads((R / "summary_revision.json").read_text())
    W = json.loads((R / "summary_fwer.json").read_text())
    L = load_ladder("baseline")
    S = json.loads((R / "summary.json").read_text())
    surv = ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1", "ABVOL"]

    # ---- Table: additional checks ------------------------------------------------
    def row(name, crit, tvals, survivors):
        others = [lab(k) for k in survivors if k not in surv]
        cells = " & ".join(fmt(tvals[k]) for k in surv)
        return f"{name} & {fmt(crit)} & {cells} & {len(survivors)} & {', '.join(others) if others else '--'} \\\\"
    base_t = {k: float(L.loc[k, "t"]) for k in ORDER}
    base_surv = [k for k in ORDER if bool(L.loc[k, "R6_rw"])]
    lines = [r"\begin{tabular}{l r rrrrr r l}", r"\toprule",
             r" & RW & \multicolumn{5}{c}{$t$-statistic} & & \\", r"\cmidrule(lr){3-7}",
             r"Specification & crit. & HIGH52 & MOM6\_1 & RESMOM & MOM12\_1 & ABVOL & RW & Other \\", r"\midrule",
             row("Baseline", S["baseline"]["rw_crit_first"], base_t, base_surv),
             r"\addlinespace[2pt]\multicolumn{9}{l}{\emph{Terminal events and suspensions}} \\"]
    B = V["bounds"]
    for key, name in (("dl30", r"Delisting return $-30$\%"), ("dl50", r"Delisting return $-50$\%"),
                      ("dl100", r"Delisting return $-100$\%"), ("adverse", r"One-sided: $-100$\% long, 0 short"),
                      ("adverse2", r"Two-sided: $-100$\% long, $+50$\% short"),
                      ("hold", "Suspensions held to resumption")):
        lines.append(row(name, B[key]["crit"], B[key]["t"], B[key]["survivors"]))
    lines.append(r"\addlinespace[2pt]\multicolumn{9}{l}{\emph{Sample and definitions}} \\")
    lines.append(row("Without receipt histories", V["noDR"]["crit"], V["noDR"]["t"], V["noDR"]["survivors"]))
    tf = V["twofactor"]["ladder"]
    lines.append(row("RESMOM, IVOL two-factor", tf["crit"], tf["t"], tf["survivors"]))
    lines.append(r"\addlinespace[2pt]\multicolumn{9}{l}{\emph{Size of the test}} \\")
    cal = W["calibrated"]
    lines.append(row(f"Calibrated RW, $\\alpha^\\ast={cal['alpha_star']:.2f}$", cal["crit_first"], base_t, cal["survivors"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_revision_checks.tex").write_text("\n".join(lines))

    # ---- Table: implementability -----------------------------------------------------
    I_ = V["implement"]
    rows_k = ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1", "ABVOL", "STR"]
    name = lambda k: lab(k) if k != "STR" else r"STR$^{\dagger}$"  # noqa: E731
    la = [r"\begin{tabular}{l rrr r r r r rr}", r"\toprule",
          r"\multicolumn{10}{l}{\emph{Panel A. Legs and short selling}} \\",
          r" & & \multicolumn{2}{c}{Leg vs.\ all sorted} & & Break-even & Net, & Net, & \multicolumn{2}{c}{Net, shorts only in} \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){9-10}",
          r" & Gross & Long & Short & Net & fee (\% p.a.) & fee 10\% & 2022 ban & U50 & Index$^{\ddagger}$ \\", r"\midrule"]
    for k in rows_k:
        r = I_[k]
        cols = [r["gross"], r["long_ex"], r["short_ex"], r["net"], None, r["net_fee10"], r["ban_longonly_net"],
                r["ok_U50_net"], r["ok_IMOEX_net"]]
        m = [(_mt(c)[0] if c is not None else fmt(100 * r["be_borrow_pa"], 1)) for c in cols]
        t = [(_mt(c)[1] if c is not None else "") for c in cols]
        la.append(f"{name(k)} & " + " & ".join(m) + r" \\")
        la.append(" & " + " & ".join(t) + r" \\[1pt]")
    la += [r"\bottomrule", r"\end{tabular}"]
    lb = [r"\begin{tabular}{l rr rr cc cc}", r"\toprule",
          r"\multicolumn{9}{l}{\emph{Panel B. Price impact (illustrative)}} \\",
          r" & \multicolumn{4}{c}{Net return} & \multicolumn{4}{c}{Break-even size (RUB m per leg)} \\",
          r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
          r" & \multicolumn{2}{c}{RUB 0.1bn per leg} & \multicolumn{2}{c}{RUB 1bn per leg} & \multicolumn{2}{c}{$Y=1$} & \multicolumn{2}{c}{$Y=0.5$} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
          r" & 2012--26 & 2022--26 & 2012--26 & 2022--26 & 2012--26 & 2022--26 & 2012--26 & 2022--26 \\", r"\midrule"]
    for k in rows_k:
        r = I_[k]
        c1, c2 = r["capacity"]["100000000.0"], r["capacity"]["1000000000.0"]
        m = [fmt(100 * c1["net"][0]), fmt(100 * c1["net_post"][0]), fmt(100 * c2["net"][0]), fmt(100 * c2["net_post"][0])]
        t = ["(" + fmt(c1["net"][1]) + ")", "(" + fmt(c1["net_post"][1]) + ")", "(" + fmt(c2["net"][1]) + ")", "(" + fmt(c2["net_post"][1]) + ")"]
        be = [_rub(r["be_aum"]), _rub(r["be_aum_post"]), _rub(r["be_aum_half"]), _rub(r["be_aum_post_half"])]
        lb.append(f"{name(k)} & " + " & ".join(m) + " & " + " & ".join(be) + r" \\")
        lb.append(" & " + " & ".join(t) + r" & & & & \\[1pt]")
    lb += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_implement_a.tex").write_text("\n".join(la))
    (TAB / "tab_implement_b.tex").write_text("\n".join(lb))

    # ---- macros ------------------------------------------------------------------------
    T = V["term"]
    num("TermCharged", int(T["charged"]))
    num("TermChargedPartial", int(T["charged_partial"]))
    num("TermChargedNone", int(T["charged_none"]))
    num("TermNoneAfter", int(T["none_after_partial"]))
    num("TermTotal", int(T["partial"] + T["none"]))
    num("TermPartial", int(T["partial"]))
    num("TermNone", int(T["none"]))
    num("TermSusp", int(T["suspended"]))
    num("TermSecids", int(T["secids"]))
    num("TermPartialMean", 100 * T["partial_mean_ret"], 0)
    num("TermPartialBelow", int(T["partial_below_m20"]))
    E = V["term_exposure"]
    num("TermHIGHlong", int(E["HIGH52"]["long"]))
    num("TermHIGHshort", int(E["HIGH52"]["short"]))
    num("TermCVVALshort", int(E["CVVAL"]["short"]))
    num("TermCVVALtot", int(E["CVVAL"]["short"] + E["CVVAL"]["long"]))
    num("AdvTwoTRESMOM", B["adverse2"]["t"]["RESMOM"])
    num("AdvTwoTABVOL", B["adverse2"]["t"]["ABVOL"])
    num("AdvTwoCrit", B["adverse2"]["crit"])
    NUM["AdvTwoCount"] = str(len([k for k in B["adverse2"]["survivors"] if k in surv]))
    num("AdvTRESMOM", B["adverse"]["t"]["RESMOM"])
    num("TermRESMOMlong", int(E["RESMOM"]["long"]))
    num("TermRESMOMshort", int(E["RESMOM"]["short"]))
    nshortmore = sum(E[k]["short"] > E[k]["long"] for k in ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1"])
    WORDS0 = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}
    NUM["TermShortMoreWord"] = WORDS0[int(nshortmore)]
    for k in ["MOM6_1", "MOM12_1"]:
        num(f"SpanTwoT{k.replace('_', '')}", V["spanning_h52_resmom"][k]["t"])
    num("AdvTABVOL", B["adverse"]["t"]["ABVOL"])
    num("AdvCrit", B["adverse"]["crit"])
    num("DRmax", int(V["noDR"]["u100_dr_max_per_month"]))
    NUM["DRmaxWord"] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                        9: "nine"}.get(int(V["noDR"]["u100_dr_max_per_month"]), str(int(V["noDR"]["u100_dr_max_per_month"])))
    num("DRstockMonths", int(V["noDR"]["u100_dr_stock_months"]))
    num("IdxMissMax", int(V["index_missing"]["max"]))
    WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    NUM["IdxMissMaxWord"] = WORDS.get(int(V["index_missing"]["max"]), str(int(V["index_missing"]["max"])))
    NUM["TermSuspWord"] = WORDS.get(int(T["suspended"]), str(int(T["suspended"])))
    NUM["TermChargedNoneWord"] = WORDS.get(int(T["charged_none"]), str(int(T["charged_none"])))
    num("IdxMissBases", int(V["index_missing"]["bases_with_missing"]))
    num("IdxBases", int(V["index_missing"]["bases"]))
    num("TwoFtRESMOM", V["twofactor"]["t2f"]["RESMOM"])
    num("TwoFtIVOL", V["twofactor"]["t2f"]["IVOL"])
    num("TwoFrankRESMOM", V["twofactor"]["rank_corr"]["RESMOM"])
    num("TwoFrankIVOL", V["twofactor"]["rank_corr"]["IVOL"])
    num("SMBmean", 100 * V["smb"]["mean"])
    num("SMBt", V["smb"]["t"])
    # FWER
    for des, tag in (("parametric", "Par"), ("bootstrap", "Boot")):
        w = W[des]
        num(f"FWrw{tag}", 100 * w["rw_rate_05"]["rate"], 1)
        num(f"FWrw{tag}Lo", 100 * w["rw_rate_05"]["ci"][0], 1)
        num(f"FWrw{tag}Hi", 100 * w["rw_rate_05"]["ci"][1], 1)
        num(f"FWholm{tag}", 100 * w["holm"]["rate"], 1)
        num(f"FWtthree{tag}", 100 * w["t3"]["rate"], 1)
        num(f"FWalpha{tag}", w["alpha_star"], 3)
    NUM["FWreps"] = f"{int(W['parametric']['reps']):,}".replace(",", "{,}")
    num("FWalphaStar", cal["alpha_star"], 2)
    num("FWcritStar", cal["crit_first"])
    num("FWpAbvol", cal["p_rw"]["ABVOL"], 3)
    # regimes
    G = pd.read_csv(R / "regimes.csv", index_col=0)
    z = G.loc["ZOO_EW"]
    se = z["delta"] / z["t_delta"]
    num("RegZooLo", 100 * (z["delta"] - 1.96 * se))
    num("RegZooHi", 100 * (z["delta"] + 1.96 * se))
    num("RegZooMDE", 100 * 2.8016 * se)
    mo = V["regime_mom4"]
    num("RegMomPre", 100 * mo["pre"])
    num("RegMomPost", 100 * mo["post"])
    num("RegMomLo", 100 * mo["lo"])
    num("RegMomHi", 100 * mo["hi"])
    num("RegMomT", mo["t"])
    # redundancy
    num("SurvCorrMin", V["surv_corr_mom_min"])
    num("SurvCorrMax", V["surv_corr_mom_max"])
    num("SurvEigShare", 100 * V["surv_mom_eig1_share"], 0)
    ab = max(V["surv_corr"]["ABVOL"][k] for k in ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1"])
    num("SurvCorrAbvolMax", ab)
    for k in ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1", "ABVOL"]:
        num(f"SpanT{k.replace('_', '')}", V["spanning"][k]["others"]["t"])
    num("CompMean", 100 * V["mom_composite"]["mean"])
    num("CompT", V["mom_composite"]["t"])
    num("CompSR", V["mom_composite"]["sr_ann"])
    # implementability
    moms = ["HIGH52", "MOM6_1", "RESMOM", "MOM12_1"]
    num("ImpLongHIGH", 100 * I_["HIGH52"]["long_ex"][0])
    num("ImpLongTHIGH", I_["HIGH52"]["long_ex"][1])
    num("ImpShortHIGH", 100 * I_["HIGH52"]["short_ex"][0])
    num("ImpShortTHIGH", I_["HIGH52"]["short_ex"][1])
    sh = [I_[k]["share_long"] for k in moms]
    num("ImpShareMin", 100 * min(sh), 0)
    num("ImpShareMax", 100 * max(sh), 0)
    be = [I_[k]["be_borrow_pa"] for k in moms]
    num("BEfeeMin", 100 * min(be), 0)
    num("BEfeeMax", 100 * max(be), 0)
    for k in moms + ["ABVOL", "STR"]:
        kk = k.replace("_", "")
        num(f"BEfee{kk}", 100 * I_[k]["be_borrow_pa"], 0)
        num(f"FeeTenT{kk}", I_[k]["net_fee10"][1])
        num(f"BanT{kk}", I_[k]["ban_longonly_net"][1])
        num(f"UfiftyT{kk}", I_[k]["ok_U50_net"][1])
        num(f"IdxT{kk}", I_[k]["ok_IMOEX_net"][1])
    for nm, key in (("Ufifty", "ok_U50_net"), ("Idx", "ok_IMOEX_net"), ("Ban", "ban_longonly_net")):
        tv = [I_[k][key][1] for k in moms]
        num(f"{nm}Tmin", min(tv))
        num(f"{nm}Tmax", max(tv))
    num("IdxFallbackABVOL", int(I_["ABVOL"]["ok_IMOEX_fallback_months"]))
    tw = [I_[k]["net_idx_window"][1] for k in moms]
    num("IdxWinTmin", min(tw))
    num("IdxWinTmax", max(tw))
    bef = [I_[k]["be_aum"] for k in moms]
    beh = [I_[k]["be_aum_half"] for k in moms]
    bep = [I_[k]["be_aum_post"] for k in moms]
    bepf = [I_[k]["be_aum_post_half"] for k in moms]
    num("CapFullMin", min(bef) / 1e6, 0)
    num("CapFullMax", max(bef) / 1e6, 0)
    num("CapHalfMin", min(beh) / 1e6, 0)
    num("CapHalfMax", max(beh) / 1e6, 0)
    num("CapPostMin", min(bep) / 1e6, 0)
    num("CapPostMax", max(bep) / 1e9, 1)
    num("CapPostHalfMin", min(bepf) / 1e6, 0)
    num("CapPostHalfMax", max(bepf) / 1e9, 1)
    NUM["ImpactZeroMonths"] = str(max(I_[k]["impact_zero_months"] for k in rows_k))
    adv = V["adv_median_by_year"]
    early = [adv[str(y)] for y in range(2012, 2016)]
    late = [adv[str(y)] for y in range(2023, 2027)]
    num("ADVearlyMin", min(early) / 1e6, 0)
    num("ADVearlyMax", max(early) / 1e6, 0)
    num("ADVlateMin", min(late) / 1e6, 0)
    num("ADVlateMax", max(late) / 1e6, 0)


def num_up(name: str, value: float, d: int = 2):
    """Upper bound rounded up, for statements of the form 'by at most x'."""
    NUM[name] = f"{np.ceil(value * 10**d - 1e-9) / 10**d:.{d}f}"


def data_checks_macros():
    """Numbers for the data-construction checks of the second revision (Appendix B)."""
    from moexzoo import data as D
    S = json.loads((R / "summary_data_sensitivity.json").read_text())
    base_surv = S["baseline"]["survivors"]
    for v, tag in (("divcal", "Divcal"), ("divcal_dedup", "Dedup"), ("yahoo_early", "YahooEarly"),
                   ("yahoo_late", "YahooLate")):
        x = S[v]
        num(f"{tag}Changed", int(x["u100_holding_returns_changed"]))
        num_up(f"{tag}MaxDt", x["max_abs_dt"])
        NUM[f"{tag}Same"] = "the same" if x["survivors"] == base_surv else "different"
    num("DedupOnlyChanged", int(S["dedup_vs_divcal"]["u100_holding_returns_changed"]))
    num_up("DedupOnlyMaxDt", S["dedup_vs_divcal"]["max_abs_dt"])
    num("YahooMaxDt", max(S["yahoo_early"]["max_abs_dt"], S["yahoo_late"]["max_abs_dt"]))
    bt = pd.Series(S["baseline"]["t"])
    dts = [pd.Series(S[v]["t"]) - bt for v in ("yahoo_early", "yahoo_late")]
    num_up("YahooMaxDtSurv", max(float(d[base_surv].abs().max()) for d in dts))
    Bv = json.loads((R / "summary_board_volume.json").read_text())
    num("BoardN", int(Bv["n_secids"]))
    vr = Bv["value_ratio_pct"]
    num("BoardValLo", min(vr[k] for k in ("SBER", "LKOH", "VTBR")), 0)
    num("BoardValHi", max(vr[k] for k in ("SBER", "LKOH", "VTBR")), 0)
    num("BoardCtlLo", min(vr[k] for k in ("GAZP", "GMKN")), 0)
    num("BoardCtlHi", max(vr[k] for k in ("GAZP", "GMKN")), 0)
    num("BoardStockMonths", int(Bv["n_u100_masked"]))
    num("BoardTABVOL", Bv["t_masked"]["ABVOL"])
    num("BoardPrwABVOL", Bv["masked"]["p_rw"]["ABVOL"], 3)
    num_up("BoardMaxDt", Bv["max_abs_dt"])
    NUM["BoardSame"] = "the same" if Bv["masked"]["survivors"] == base_surv else "different"
    NUM["BoardCountsSame"] = "unchanged" if Bv["masked"]["counts"] == Bv["baseline"]["counts"] else "changed"
    kmax = pd.concat([d.abs() for d in dts], axis=1).max(axis=1).idxmax()
    NUM["YahooMaxDtId"] = LABEL.get(kmax, kmax)
    num("YahooMaxDtBaseT", float(bt[kmax]))
    same_counts = all(S[v]["counts"] == S["baseline"]["counts"]
                      for v in ("divcal", "divcal_dedup", "yahoo_early", "yahoo_late"))
    NUM["DataCountsSame"] = "unchanged" if same_counts else "changed"
    num("DedupDropped", int(S["divcal_dedup"]["div_dedup_dropped_all"]))
    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    u100_ever = set(p.loc[p["U100"], "secid"])
    u100_hold = set(zip(p.loc[p["U100"], "month"] + 1, p.loc[p["U100"], "secid"]))
    # payments that share a record date (securities with price data, as in the pipeline)
    g_all = g_pairs = g_trip = g_u = drop_u = big_u = 0
    # payments credited under the plan rule, by source
    n_all = n_yah = n_all_u = n_yah_u = 0
    for f in sorted((C.MM / "dividends").glob("*.csv")):
        sec = f.stem
        if not (C.MM / "prices_iss" / f"{sec}.csv").exists():
            continue
        d = D.load_dividends(sec)
        d = d[d["currency"].astype(str) == "RUB"]
        sz = d.groupby("registry_close").size()
        sz = sz[sz > 1]
        g_all += len(sz)
        g_pairs += int((sz == 2).sum())
        g_trip += int((sz == 3).sum())
        if sec in u100_ever:
            g_u += len(sz)
            drop_u += int((sz - 1).sum())
            big_u += int((sz > 2).sum())
        px = D.load_prices(sec)
        if px.empty:
            continue
        idx = px.index
        for _, r in d.iterrows():
            src = str(r.get("source", ""))
            reg = pd.Timestamp(r["registry_close"])
            pos = int(idx.searchsorted(reg, side="right")) - 1 - D.settlement_lag(src, reg) + 1
            if pos <= 0 or pos >= len(idx):
                continue
            ex = idx[pos]
            if not (reg.to_period("M") - 1 <= ex.to_period("M") <= reg.to_period("M") + 1):
                continue
            y = int(src == C.EX_DATE_SOURCE)
            n_all += 1
            n_yah += y
            if (ex.to_period("M"), sec) in u100_hold:
                n_all_u += 1
                n_yah_u += y
    assert g_all == g_pairs + g_trip and big_u == 0, (g_all, g_pairs, g_trip, big_u)
    assert g_pairs + 2 * g_trip == int(S["divcal_dedup"]["div_dedup_dropped_all"])  # matches the variant's drops
    num("DivGroupsAll", g_all)
    num("DivGroupsPairsAll", g_pairs)
    num("DivGroupsTriples", g_trip)
    num("DivPairs", g_u)  # all of them pairs (asserted above)
    num("DedupDroppedU", drop_u)
    num("DivYahooShare", 100.0 * n_yah / n_all, 0)
    num("DivYahooShareU", 100.0 * n_yah_u / n_all_u, 0)
    m = pd.read_parquet(C.INTERIM / "monthly.parquet")
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    ev = ev_pit = 0
    for f in (C.MM / "splits").glob("*.csv"):
        g = m[m["secid"] == f.stem].set_index("month").sort_index()
        for dte, _ in D.load_splits(f.stem):
            b, a = g[g.index < dte.to_period("M")], g[g.index >= dte.to_period("M")]
            if len(b) and len(a) and pd.notna(b.iloc[-1]["shares_adj"]) and pd.notna(a.iloc[0]["shares_adj"]):
                ev += 1
                ev_pit += int(b.iloc[-1]["shares_src"] == "index_base_pit" and a.iloc[0]["shares_src"] == "index_base_pit")
    num("NsplitEvents", ev)
    num("NsplitPit", ev_pit)


def main():
    mechanism_macros()
    df = table_sample()
    table_definitions()
    table_baseline()
    table_ladder_counts()
    table_regimes()
    table_costs()
    table_fm()
    table_power()
    table_robust_t()
    table_eb()
    table_str_turnover()
    revision_tables()
    data_checks_macros()
    fig_sample(df)
    fig_ladder()
    fig_boot()
    fig_cumret()
    fig_regimes()
    fig_power()
    fig_eb()
    fig_corr()
    lines = ["% Auto-generated by scripts/06_tables_figures.py -- do not edit by hand.",
             r"\makeatletter",
             r"\newcommand{\val}[1]{\@ifundefined{val@#1}{\textbf{??#1??}}{\csname val@#1\endcsname}}",
             r"\makeatother"]
    for k, v in sorted(NUM.items()):
        lines.append("\\expandafter\\def\\csname val@" + k + "\\endcsname{" + v + "}")
    (C.ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"tables: {len(list(TAB.glob('*.tex')))}  figures: {len(list(FIG.glob('*.pdf')))}  macros: {len(NUM)}")


if __name__ == "__main__":
    main()
