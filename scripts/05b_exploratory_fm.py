"""Exploratory (post-results) Fama–MacBeth specifications and the STR turnover-difference test.
Outputs results/exploratory_fm.csv and results/exploratory_str_turnover_diff.csv"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402
from moexzoo import chars as CH  # noqa: E402
from moexzoo import inference as I  # noqa: E402
from moexzoo import portfolios as PF  # noqa: E402

SPECS = {"A_one_per_cluster": ["MOM12_1", "STR", "SEAS", "BETA", "DBETA", "SIZE", "DY"],
         "B_survivors_plus_STR": ["HIGH52", "MOM12_1", "MOM6_1", "RESMOM", "ABVOL", "STR"]}


def main():
    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    out = []
    for name, ks in SPECS.items():
        rows = []
        for t, g in p.groupby("month", sort=True):
            hm = t + 1
            if hm < pd.Period(C.SAMPLE_START, "M") or hm > pd.Period(C.SAMPLE_END, "M"):
                continue
            g = g[g["U100"]]
            y = g["R_next"].fillna(0.0).values
            Z = []
            for k in ks:
                x = g[k].where(g[k] > 0) if k == "DY" else g[k]
                r = x.rank(method="average"); n = r.notna().sum()
                Z.append((CH.SIGN[k] * ((r - 1) / (n - 1) - 0.5)).fillna(0.0).values)
            X = np.column_stack([np.ones(len(y))] + Z)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            rows.append(pd.Series(b[1:], index=ks, name=hm))
        ts = I.tstats(pd.DataFrame(rows))
        ts["spec"] = name
        out.append(ts[["mean", "t", "n", "spec"]])
    pd.concat(out).to_csv(C.RESULTS / "exploratory_fm.csv")
    print(pd.concat(out).round(4))
    # STR by turnover: high-minus-low turnover difference in continuation
    rows = {}
    for t, g in p.groupby("month", sort=True):
        hm = t + 1
        if hm < pd.Period(C.SAMPLE_START, "M") or hm > pd.Period(C.SAMPLE_END, "M"):
            continue
        g = g[g["UALL"]].dropna(subset=["STR", "TURN"]).set_index("secid")
        if len(g) < 60:
            continue
        tg = PF._groups(g["TURN"], 3)
        res = {}
        for q in (1, 2, 3):
            sub = g.loc[tg.index[tg == q]]
            sg = PF._groups(sub["STR"], 5)
            res[q] = sub.loc[sg.index[sg == 5], "R_next"].fillna(0).mean() - sub.loc[sg.index[sg == 1], "R_next"].fillna(0).mean()
        rows[hm] = {"low_turn": res[1], "mid_turn": res[2], "high_turn": res[3], "high_minus_low": res[3] - res[1]}
    D = pd.DataFrame(rows).T
    ts = I.tstats(D)
    ts.to_csv(C.RESULTS / "exploratory_str_turnover_diff.csv")
    print(ts[["mean", "t", "n"]].round(4))


if __name__ == "__main__":
    main()
