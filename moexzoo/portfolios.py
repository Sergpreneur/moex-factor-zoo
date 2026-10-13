"""Characteristic-sorted long–short portfolios (pre-analysis plan §5, §7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class LSResult:
    ret: pd.Series  # long–short return indexed by holding month (Period)
    long: pd.Series
    short: pd.Series
    n_long: pd.Series
    n_short: pd.Series
    turnover: pd.Series  # two-sided turnover of the long–short position at formation (sum of both legs)
    cost: pd.Series  # cost charged in the holding month
    n_cash: pd.Series  # held names with no trade in the holding month (earn 0)


def _groups(x: pd.Series, n_groups: int) -> pd.Series:
    """Quantile groups 1..n by rank; ties broken by SECID (index order is SECID-sorted)."""
    x = x.dropna()
    order = sorted(x.index, key=lambda s: (x[s], s))
    n = len(order)
    g = {s: int(np.floor(i * n_groups / n)) + 1 for i, s in enumerate(order)}
    return pd.Series(g)


def _leg(weights: pd.Series, rets: pd.Series) -> tuple[float, int]:
    r = rets.reindex(weights.index)
    ncash = int(r.isna().sum())
    r = r.fillna(0.0)  # no trade in the holding month -> cash
    return float((weights * r).sum()), ncash


def _weights(names: pd.Index, caps: pd.Series | None, weighting: str, cap_limit: float | None) -> pd.Series:
    if weighting == "EW":
        return pd.Series(1.0 / len(names), index=names)
    c = caps.reindex(names).dropna()
    c = c[c > 0]
    if cap_limit is not None:
        c = c.clip(upper=cap_limit)
    return c / c.sum()


def long_short(panel: pd.DataFrame, char: str, sign: int, *, universe: str = "U100", weighting: str = "EW",
               n_groups: int = C.N_GROUPS, ret_col: str = "R_next", min_n: int = C.MIN_STOCKS_PER_SORT,
               with_costs: bool = True, cost_mult: float = 1.0, payers_only: bool | None = None) -> LSResult:
    if payers_only is None:
        payers_only = char == "DY"
    rows = {}
    prev_w = None  # previous month's (long, short) weights after drift
    for t, g in panel.groupby("month", sort=True):
        g = g[g[universe]].set_index("secid")
        x = g[char]
        if payers_only:
            x = x.where(x > 0)
        x = x.dropna()
        hm = t + 1
        if len(x) < min_n:
            rows[hm] = dict(ret=np.nan, long=np.nan, short=np.nan, nl=0, ns=0, to=np.nan, cost=np.nan, ncash=0)
            prev_w = None
            continue
        grp = _groups(x, n_groups)
        hi, lo = grp.index[grp == n_groups], grp.index[grp == 1]
        if sign < 0:
            hi, lo = lo, hi
        cap_limit = None
        if weighting == "VW":
            capu = g["cap"].dropna()
            cap_limit = capu.quantile(C.CAP_WEIGHT_PERCENTILE) if len(capu) else None
        wl = _weights(hi, g["cap"], weighting, cap_limit)
        ws = _weights(lo, g["cap"], weighting, cap_limit)
        if len(wl) == 0 or len(ws) == 0:
            rows[hm] = dict(ret=np.nan, long=np.nan, short=np.nan, nl=0, ns=0, to=np.nan, cost=np.nan, ncash=0)
            prev_w = None
            continue
        rl, cl = _leg(wl, g[ret_col])
        rs, cs = _leg(ws, g[ret_col])
        # turnover vs drifted previous weights (each leg separately; two legs summed)
        if prev_w is None:
            to_l, to_s = 1.0, 1.0
            dl = wl
            ds = ws
            prev_l = pd.Series(dtype=float)
            prev_s = pd.Series(dtype=float)
        else:
            prev_l, prev_s = prev_w
        idx_l = wl.index.union(prev_l.index)
        idx_s = ws.index.union(prev_s.index)
        tr_l = (wl.reindex(idx_l, fill_value=0.0) - prev_l.reindex(idx_l, fill_value=0.0)).abs()
        tr_s = (ws.reindex(idx_s, fill_value=0.0) - prev_s.reindex(idx_s, fill_value=0.0)).abs()
        cost = np.nan
        if with_costs:
            cst = g["cost"]
            cm = cst.median()
            c_l = cst.reindex(idx_l).fillna(cm if pd.notna(cm) else 0.005)
            c_s = cst.reindex(idx_s).fillna(cm if pd.notna(cm) else 0.005)
            cost = cost_mult * (float((tr_l * c_l).sum()) + float((tr_s * c_s).sum()))
        to = float(tr_l.sum() + tr_s.sum())
        rows[hm] = dict(ret=rl - rs, long=rl, short=rs, nl=len(wl), ns=len(ws), to=to, cost=cost, ncash=cl + cs)
        # drift weights through the holding month for the next rebalance
        r_l = g[ret_col].reindex(wl.index).fillna(0.0)
        r_s = g[ret_col].reindex(ws.index).fillna(0.0)
        dl = wl * (1 + r_l)
        ds = ws * (1 + r_s)
        prev_w = (dl / dl.sum() if dl.sum() > 0 else wl, ds / ds.sum() if ds.sum() > 0 else ws)
    df = pd.DataFrame(rows).T
    df.index = pd.PeriodIndex(df.index, freq="M")
    df = df.sort_index()
    return LSResult(ret=df["ret"].astype(float), long=df["long"].astype(float), short=df["short"].astype(float),
                    n_long=df["nl"].astype(float), n_short=df["ns"].astype(float), turnover=df["to"].astype(float),
                    cost=df["cost"].astype(float), n_cash=df["ncash"].astype(float))
