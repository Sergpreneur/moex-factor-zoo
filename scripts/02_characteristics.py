"""Universes, characteristics, costs and holding-period returns for every formation month.

Output: data/processed/panel.parquet — one row per (formation month t, secid) for shares in UALL,
with universe flags, the 25 raw characteristics, SIZE_PIT/TURN_PIT, one-way cost estimate, cap at t
(and whether point-in-time), and holding returns over month t+1 (total, price-only, 1-session-lag).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402
from moexzoo import chars as CH  # noqa: E402
from moexzoo import universe as UN  # noqa: E402


def main() -> None:
    t0 = time.time()
    P = CH.Panel()
    U = UN.build(P.monthly, P.sessions)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    U.to_parquet(C.PROCESSED / "universes.parquet")
    print("universes built", U.groupby("month")[["UALL", "U100", "U50", "U100ORD"]].sum().describe().round(1).to_string())

    # holding returns
    R = P.R
    Rx = P.monthly.pivot(index="month", columns="secid", values="retx").sort_index().reindex(R.index)
    # one-session-lag entry: TRI at first trade of month t+1 if within its first 3 sessions
    tri = P.tri
    month_of = P.month_of
    first3 = pd.Series(np.arange(len(P.sessions)), index=P.sessions).groupby(month_of).transform(lambda x: x - x.min()) < 3
    tri_first = tri.where(np.broadcast_to(first3.values[:, None], tri.shape)).groupby(month_of).apply(lambda g: g.bfill().iloc[0])
    tri_last = tri.groupby(month_of).last()
    traded_any = P.traded.groupby(month_of).any()
    R_lag = (tri_last / tri_first - 1.0).where(traded_any)
    R_lag.index = pd.PeriodIndex(R_lag.index, freq="M")
    R_lag = R_lag.reindex(R.index)

    cap_pit = P.cap.where(P.cap_src == "index_base_pit")
    form_months = pd.period_range(C.FIRST_FORMATION, pd.Period(C.SAMPLE_END, "M") - 1, freq="M")
    out = []
    for t in form_months:
        uni = U[U["month"] == t].set_index("secid")
        if uni.empty:
            continue
        ch = CH.characteristics_at(P, t).reindex(uni.index)
        # point-in-time-only size and turnover (robustness j)
        it = P.R.index.get_loc(t)
        cp = cap_pit.iloc[it].reindex(uni.index)
        ch["SIZE_PIT"] = np.log(cp.where(cp > 0))
        capw = cap_pit.iloc[it - 11: it + 1].reindex(columns=uni.index)
        vw = P.value_m.iloc[it - 11: it + 1].reindex(columns=uni.index).fillna(0.0)
        ch["TURN_PIT"] = (vw.sum() / capw.mean()).where((capw.notna().sum() >= 6) & cp.notna())
        cost = CH.costs_at(P, t).reindex(uni.index)
        tn = t + 1
        df = uni.join(ch)
        df["cost"] = cost
        df["cap"] = P.cap.loc[t].reindex(uni.index)
        df["cap_pit"] = cp.notna()
        df["R_next"] = R.loc[tn].reindex(uni.index) if tn in R.index else np.nan
        df["Rx_next"] = Rx.loc[tn].reindex(uni.index) if tn in Rx.index else np.nan
        df["Rlag_next"] = R_lag.loc[tn].reindex(uni.index) if tn in R_lag.index else np.nan
        df["month"] = t
        out.append(df.reset_index())
    panel = pd.concat(out, ignore_index=True)
    panel.to_parquet(C.PROCESSED / "panel.parquet")
    print(f"panel rows {len(panel)}  months {panel['month'].nunique()}  time {time.time() - t0:.0f}s")
    cov = panel[panel["U100"]].groupby("month")[CH.CHAR_IDS].count().mean().round(1)
    print("mean # U100 names with each characteristic:\n", cov.to_string())


if __name__ == "__main__":
    main()
