"""Monthly investable universes (pre-analysis plan §3)."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from . import config as C


def preferred_secids(tickers: dict) -> set[str]:
    pat = re.compile(r"(\bап\b|-п\b|pref|Pref|прив|\bаp\b)")
    out = set()
    for s, v in tickers.items():
        txt = " | ".join(v.get("aliases", []) or []) + " | " + str(v.get("canonical", ""))
        if pat.search(txt):
            out.add(s)
    return out


def rename_edges(tickers: dict) -> list[tuple[str, str, pd.Period]]:
    edges = []
    for succ, e in tickers.items():
        for h in e.get("history") or ():
            if h.get("source") != "iss_changeover":
                continue
            pred, ren = h.get("prev_ticker"), h.get("renamed")
            if pred and ren and pred != succ:
                edges.append((pred, succ, pd.Period(ren, freq="M")))
    return edges


def build(monthly: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    tickers = json.loads((C.MM / "tickers.json").read_text())
    m = monthly.copy()
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    sess_per_month = pd.Series(1, index=sessions).groupby(sessions.to_period("M")).sum()
    month_end = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()

    nd = m.pivot(index="month", columns="secid", values="ndays").sort_index()
    gap = m.pivot(index="month", columns="secid", values="gap_to_end").sort_index()
    val = m.pivot(index="month", columns="secid", values="value_m").sort_index()
    allm = pd.period_range(nd.index.min(), nd.index.max(), freq="M")
    nd, gap, val = nd.reindex(allm), gap.reindex(allm), val.reindex(allm)
    med12 = val.rolling(C.HISTORY_MONTHS_FOR_LIQ, min_periods=C.HISTORY_MONTHS_FOR_LIQ).median()

    need = pd.Series({p: min(C.MIN_TRADING_DAYS_IN_MONTH, math.ceil(0.5 * sess_per_month.get(p, 0)))
                      for p in allm})
    active = nd.ge(need, axis=0) & (gap <= C.LAST_TRADE_WITHIN_DAYS - 1) & med12.notna()
    # delisting flag from the dictionary
    for s in active.columns:
        da = tickers.get(s, {}).get("delisted_after")
        if da:
            dts = pd.Timestamp(da)
            me = month_end.reindex(allm)
            active.loc[(me >= dts).values, s] = False
    # one issuer, one slot across ISS renames
    for pred, succ, ren in rename_edges(tickers):
        if pred in active.columns and succ in active.columns:
            before = active.index < ren
            # before the rename keep the predecessor (if it trades), else the successor's stitched copy
            both = active[pred] & active[succ]
            active.loc[before & both.values, succ] = False
            active.loc[~before, pred] = False

    liq = med12.where(active)
    rows = []
    prefs = preferred_secids(tickers)
    for p in allm:
        a = liq.loc[p].dropna()
        if a.empty:
            continue
        order = sorted(a.index, key=lambda s: (-a[s], s))
        rank = {s: i + 1 for i, s in enumerate(order)}
        u100 = set(order[: C.TOP_N_BASELINE])
        u50 = set(order[: C.TOP_N_LARGE])
        ords_in = {s for s in u100 if s not in prefs}
        u100ord = {s for s in u100 if not (s in prefs and s[:-1] in ords_in)}
        for s in order:
            rows.append((p, s, rank[s], a[s], True, s in u100, s in u50, s in u100ord))
    U = pd.DataFrame(rows, columns=["month", "secid", "liq_rank", "med12_value", "UALL", "U100", "U50", "U100ORD"])
    return U
