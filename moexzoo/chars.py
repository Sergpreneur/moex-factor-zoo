"""The 25 characteristics of the pre-analysis plan (§4), computed at month-ends.

Every function takes data observed up to and including the last session of
formation month t and nothing later. The integrity test checks this
by perturbing data after t and verifying that characteristics at t do not move
(tests/test_pipeline.py::test_no_lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CHARS = [
    # id, cluster, predicted sign
    ("MOM12_1", "Momentum", +1),
    ("MOM6_1", "Momentum", +1),
    ("MOM12_7", "Momentum", +1),
    ("RESMOM", "Momentum", +1),
    ("HIGH52", "Momentum", +1),
    ("STR", "Reversal", -1),
    ("LTR", "Reversal", -1),
    ("SEAS", "Seasonality", +1),
    ("BETA", "Low risk", -1),
    ("IVOL", "Low risk", -1),
    ("TVOL", "Low risk", -1),
    ("MAX", "Low risk", -1),
    ("DBETA", "Skewness/downside", +1),
    ("SKEW", "Skewness/downside", -1),
    ("ISKEW", "Skewness/downside", -1),
    ("COSKEW", "Skewness/downside", -1),
    ("SIZE", "Size/liquidity", -1),
    ("AMIHUD", "Size/liquidity", +1),
    ("DVOL", "Size/liquidity", -1),
    ("TURN", "Size/liquidity", -1),
    ("ZERO", "Size/liquidity", +1),
    ("SPREAD", "Size/liquidity", +1),
    ("CVVAL", "Size/liquidity", -1),
    ("ABVOL", "Size/liquidity", +1),
    ("DY", "Payout", +1),
]
CHAR_IDS = [c[0] for c in CHARS]
SIGN = {c[0]: c[2] for c in CHARS}
CLUSTER = {c[0]: c[1] for c in CHARS}


# ----------------------------------------------------------------------------------
# masked moment helpers (arrays T x N with NaN; x is T-vector)
# ----------------------------------------------------------------------------------

def _ols1(y: np.ndarray, x: np.ndarray):
    mask = ~np.isnan(y) & ~np.isnan(x)[:, None]
    n = mask.sum(0).astype(float)
    X = np.where(mask, x[:, None], 0.0)
    Y = np.where(mask, y, 0.0)
    sx, sy = X.sum(0), Y.sum(0)
    sxx, sxy = (X * X).sum(0), (X * Y).sum(0)
    den = n * sxx - sx**2
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (n * sxy - sx * sy) / den
        alpha = (sy - beta * sx) / n
    resid = np.where(mask, y - alpha - beta * x[:, None], np.nan)
    return beta, alpha, resid, n, mask


def _skew(a: np.ndarray, n: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nansum(a, 0) / n
        d = a - mu
        m2 = np.nansum(d**2, 0) / n
        m3 = np.nansum(d**3, 0) / n
        return m3 / m2**1.5


def _dimson(y: np.ndarray, m: np.ndarray, mlag: np.ndarray):
    """beta = b0 + b1 from y = a + b0 m + b1 mlag + e, per column with masks."""
    mask = ~np.isnan(y) & ~np.isnan(m)[:, None] & ~np.isnan(mlag)[:, None]
    n = mask.sum(0).astype(float)
    w = mask.astype(float)
    Y = np.where(mask, y, 0.0)
    x1 = m[:, None] * w
    x2 = mlag[:, None] * w
    x1 = np.nan_to_num(x1)
    x2 = np.nan_to_num(x2)
    S = np.empty((y.shape[1], 3, 3))
    S[:, 0, 0] = n
    S[:, 0, 1] = S[:, 1, 0] = x1.sum(0)
    S[:, 0, 2] = S[:, 2, 0] = x2.sum(0)
    S[:, 1, 1] = (x1 * x1).sum(0)
    S[:, 1, 2] = S[:, 2, 1] = (x1 * x2).sum(0)
    S[:, 2, 2] = (x2 * x2).sum(0)
    b = np.stack([Y.sum(0), (x1 * Y).sum(0), (x2 * Y).sum(0)], axis=1)
    out = np.full(y.shape[1], np.nan)
    ok = n >= 3
    if ok.any():
        Sok = S[ok]
        det = np.linalg.det(Sok)
        good = np.abs(det) > 1e-18
        sol = np.full((ok.sum(), 3), np.nan)
        sol[good] = np.linalg.solve(Sok[good], b[ok][good][..., None])[..., 0]
        out[ok] = sol[:, 1] + sol[:, 2]
    return out, n


def _corwin_schultz(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean two-day Corwin–Schultz spread over pairs (d, d+1) inside the window.
    Overnight adjustment of day d+1 range by the close of day d; negatives -> 0."""
    h0, l0, c0 = h[:-1], l[:-1], c[:-1]
    h1, l1 = h[1:].copy(), l[1:].copy()
    up = l1 > c0  # gap up: shift day-2 range down
    adj = np.where(up, l1 - c0, 0.0)
    h1 = h1 - np.nan_to_num(adj)
    l1 = l1 - np.nan_to_num(adj)
    dn = h1 < c0  # gap down: shift up
    adj2 = np.where(dn, c0 - h1, 0.0)
    h1 = h1 + np.nan_to_num(adj2)
    l1 = l1 + np.nan_to_num(adj2)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
        gamma = np.log(np.maximum(h0, h1) / np.minimum(l0, l1)) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isnan(s), np.nan, np.maximum(s, 0.0))
    n = (~np.isnan(s)).sum(0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(n[None, :] > 0, s, np.nan), axis=0)
    return mean, n


def _abdi_ranaldo(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monthly Abdi–Ranaldo spread: sqrt(max(4*mean[(c_t-eta_t)(c_t-eta_{t+1})], 0))."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lc = np.log(c)
        eta = 0.5 * (np.log(h) + np.log(l))
        prod = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    n = (~np.isnan(prod)).sum(0)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(n[None, :] > 0, prod, np.nan), axis=0)
    return np.sqrt(np.maximum(4.0 * m, 0.0)), n


# ----------------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------------

class Panel:
    """Holds the daily wide panels and the monthly panels needed for characteristics."""

    def __init__(self, interim=C.INTERIM):
        rd = lambda k: pd.read_parquet(interim / f"daily_{k}.parquet")  # noqa: E731
        self.close = rd("close")
        self.high = rd("high")
        self.low = rd("low")
        self.value = rd("value")
        self.ret = rd("ret")
        self.div = rd("div")
        self.tri = rd("tri")
        self.sessions = self.close.index
        self.secids = self.close.columns
        mk = pd.read_parquet(interim / "market_daily.parquet")
        gap = pd.Series(self.sessions, index=self.sessions).diff().dt.days
        mret = mk["ret"].copy()
        mret[gap > C.MAX_SESSION_GAP_DAYS] = np.nan
        self.mret = mret
        self.mret_lag = mret.shift(1)
        self.traded = self.close.notna()
        # own-trading-day price change (for ZERO): vs previous trade, regardless of gaps
        prev_close = self.close.ffill().shift(1)
        self.own_chg = (self.close / prev_close - 1.0).where(self.traded)
        first_trade = self.traded.idxmax()
        first_trade[~self.traded.any()] = pd.NaT
        self.first_trade = first_trade
        self.month_of = self.sessions.to_period("M")
        m = pd.read_parquet(interim / "monthly.parquet")
        self.monthly = m
        self.R = m.pivot(index="month", columns="secid", values="ret").sort_index()
        self.value_m = m.pivot(index="month", columns="secid", values="value_m").sort_index()
        self.cap = m.pivot(index="month", columns="secid", values="cap").sort_index()
        self.cap_src = m.pivot(index="month", columns="secid", values="shares_src").sort_index()
        mm = pd.read_parquet(interim / "market_monthly.parquet")
        mm.index = pd.PeriodIndex(mm.index, freq="M")
        self.mkt_m = mm["mkt_ret"]
        self.rf_m = mm["rf"]
        allm = pd.period_range(self.R.index.min(), self.R.index.max(), freq="M")
        self.R = self.R.reindex(allm)
        self.value_m = self.value_m.reindex(allm)
        self.cap = self.cap.reindex(allm)
        self.cap_src = self.cap_src.reindex(allm)
        self.sess_per_month = pd.Series(1, index=self.sessions).groupby(self.month_of).sum()


def characteristics_at(P: Panel, t: pd.Period) -> pd.DataFrame:
    """All 25 characteristics (raw, un-oriented) for every SECID at month-end t."""
    S = P.sessions
    in_m = np.where(P.month_of == t)[0]
    if len(in_m) == 0:
        return pd.DataFrame(index=P.secids, columns=CHAR_IDS, dtype=float)
    end = in_m[-1]
    w12 = slice(max(0, end - C.DAYS_12M + 1), end + 1)
    w1 = slice(in_m[0], end + 1)
    out = pd.DataFrame(index=P.secids, dtype=float)

    # ---------------- monthly-return based ----------------------------------------
    R = P.R
    months = R.index
    if t not in months:
        return pd.DataFrame(index=P.secids, columns=CHAR_IDS, dtype=float)
    it = months.get_loc(t)

    def block(a, b):  # months t-a .. t-b inclusive (a >= b)
        lo, hi = it - a, it - b
        if lo < 0:
            return None
        return R.iloc[lo: hi + 1]

    def cumret(blk, need):
        if blk is None:
            return pd.Series(np.nan, index=R.columns)
        cnt = blk.notna().sum()
        val = (1.0 + blk.fillna(0.0)).prod() - 1.0
        return val.where(cnt >= need)

    out["MOM12_1"] = cumret(block(11, 1), 11)
    out["MOM6_1"] = cumret(block(5, 1), 5)
    out["MOM12_7"] = cumret(block(11, 6), 6)
    out["STR"] = R.iloc[it]
    out["LTR"] = cumret(block(35, 12), 20)
    seas_rows = [it + 1 - 12 * j for j in range(1, 6) if it + 1 - 12 * j >= 0]
    seas = R.iloc[seas_rows]
    out["SEAS"] = seas.mean().where(seas.notna().sum() >= 2)

    # residual momentum
    blk = block(35, 1)
    if blk is not None:
        mk = (P.mkt_m - P.rf_m).reindex(blk.index).values
        y = blk.sub(P.rf_m.reindex(blk.index).values, axis=0).values
        beta, alpha, resid, n, mask = _ols1(y, mk)
        last11 = resid[-11:]
        full11 = (~np.isnan(last11)).sum(0) == 11
        with np.errstate(invalid="ignore", divide="ignore"):
            rm = np.nansum(last11, 0) / np.nanstd(last11, 0, ddof=1)
        out["RESMOM"] = pd.Series(np.where((n >= 24) & full11, rm, np.nan), index=R.columns)
    else:
        out["RESMOM"] = np.nan

    # ---------------- daily, 12M -------------------------------------------------
    r12 = P.ret.iloc[w12].values
    m12 = P.mret.iloc[w12].values
    ml12 = P.mret_lag.iloc[w12].values
    bD, nD = _dimson(r12, m12, ml12)
    out["BETA"] = pd.Series(np.where(nD >= C.MIN_OBS_12M, bD, np.nan), index=P.secids)

    beta, alpha, resid, n, mask = _ols1(r12, m12)
    ok12 = n >= C.MIN_OBS_12M
    # skewness of raw returns (on obs where r valid)
    nr = (~np.isnan(r12)).sum(0)
    out["SKEW"] = pd.Series(np.where(nr >= C.MIN_OBS_12M, _skew(r12, nr), np.nan), index=P.secids)
    out["ISKEW"] = pd.Series(np.where(ok12, _skew(resid, n), np.nan), index=P.secids)
    # coskewness (Harvey-Siddique) with market-model residuals, demeaned market on the same obs
    mm = np.where(mask, m12[:, None], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        em = mm - np.nanmean(mm, 0)
        num = np.nanmean(resid * em**2, 0)
        den = np.sqrt(np.nanmean(resid**2, 0)) * np.nanmean(em**2, 0)
        cosk = num / den
    out["COSKEW"] = pd.Series(np.where(ok12, cosk, np.nan), index=P.secids)
    # downside beta
    with np.errstate(invalid="ignore"):
        mbar = np.nanmean(mm, 0)
        down = mask & (mm < mbar)
    nd = down.sum(0)
    rd_ = np.where(down, r12, np.nan)
    md_ = np.where(down, mm, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.nanmean((rd_ - np.nanmean(rd_, 0)) * (md_ - np.nanmean(md_, 0)), 0)
        var = np.nanmean((md_ - np.nanmean(md_, 0)) ** 2, 0)
        db = cov / var
    out["DBETA"] = pd.Series(np.where(ok12 & (nd >= 40), db, np.nan), index=P.secids)

    # liquidity, 12M
    v12 = P.value.iloc[w12]
    tr12 = P.traded.iloc[w12]
    ntr = tr12.sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        illiq = np.abs(r12) / (v12.values / 1e6)
    illiq[(v12.values <= 0) | np.isnan(v12.values)] = np.nan
    nil = (~np.isnan(illiq)).sum(0)
    with np.errstate(invalid="ignore"):
        out["AMIHUD"] = pd.Series(np.where(nil >= C.MIN_OBS_12M, np.nanmean(illiq, 0), np.nan), index=P.secids)
    # sessions since first trade within window
    ft = P.first_trade.reindex(P.secids)
    sess12 = S[w12]
    since = np.array([(sess12 >= f).sum() if pd.notna(f) else 0 for f in ft.values])
    vfill = v12.fillna(0.0).values
    alive = (sess12.values[:, None] >= ft.values[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_val = (vfill * alive).sum(0) / since
    out["DVOL"] = pd.Series(np.where(ntr >= C.MIN_OBS_12M, np.log(mean_val), np.nan), index=P.secids)
    chg = P.own_chg.iloc[w12].values
    zero_day = alive & (~tr12.values | (np.abs(np.nan_to_num(chg, nan=1.0)) < 1e-12))
    with np.errstate(invalid="ignore", divide="ignore"):
        zr = zero_day.sum(0) / since
    out["ZERO"] = pd.Series(np.where(since >= C.MIN_OBS_12M, zr, np.nan), index=P.secids)
    c12 = P.close.iloc[w12]
    hi = c12.max()
    lastc = P.close.iloc[: end + 1].ffill().iloc[-1]
    out["HIGH52"] = (lastc / hi).where(pd.Series(ntr, index=P.secids) >= C.MIN_OBS_12M)

    # dividends over months t-11..t (sessions of those months)
    mfirst = t - 11
    sel = (P.month_of >= mfirst) & (P.month_of <= t)
    divsum = P.div.loc[sel].sum()
    dy = divsum / lastc
    out["DY"] = dy.where(dy > 0)

    # ---------------- daily, 1M ------------------------------------------------
    r1 = P.ret.iloc[w1].values
    m1 = P.mret.iloc[w1].values
    n1 = (~np.isnan(r1)).sum(0)
    with np.errstate(invalid="ignore"):
        out["TVOL"] = pd.Series(np.where(n1 >= C.MIN_OBS_1M, np.nanstd(r1, 0, ddof=1), np.nan), index=P.secids)
        out["MAX"] = pd.Series(np.where(n1 >= C.MIN_OBS_1M, np.nanmax(np.where(n1[None, :] > 0, r1, -np.inf), 0), np.nan), index=P.secids)
    b1, a1, e1, ne1, _ = _ols1(r1, m1)
    with np.errstate(invalid="ignore"):
        out["IVOL"] = pd.Series(np.where(ne1 >= C.MIN_OBS_1M, np.nanstd(e1, 0, ddof=1), np.nan), index=P.secids)
    h1 = P.high.iloc[w1].values
    l1 = P.low.iloc[w1].values
    cc1 = P.close.iloc[w1].values
    cs, ncs = _corwin_schultz(h1, l1, cc1)
    out["SPREAD"] = pd.Series(np.where(ncs >= 12, cs, np.nan), index=P.secids)

    # ---------------- monthly liquidity / size ----------------------------------
    vm = P.value_m
    first_m = P.first_trade.reindex(P.secids).dt.to_period("M")
    win = vm.iloc[it - 11: it + 1] if it >= 11 else None
    if win is not None:
        listed = pd.DataFrame({s: (win.index >= fm) if pd.notna(fm) else np.zeros(len(win), bool)
                               for s, fm in first_m.items()}, index=win.index)
        w0 = win.reindex(columns=P.secids).fillna(0.0).where(listed)
        nlist = listed.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = w0.std(ddof=1) / w0.mean()
        out["CVVAL"] = cv.where((nlist >= 12) & (w0.mean() > 0))
        # turnover = sum value / mean cap over t-11..t
        capw = P.cap.iloc[it - 11: it + 1].reindex(columns=P.secids)
        ncap = capw.notna().sum()
        out["TURN"] = (w0.sum() / capw.mean()).where(ncap >= 6)
        # abnormal volume: avg daily value in t vs avg daily value over t-12..t-1
        spm = P.sess_per_month.reindex(vm.index).values
    else:
        out["CVVAL"] = np.nan
        out["TURN"] = np.nan
    if it >= 12:
        prev = vm.iloc[it - 12: it].reindex(columns=P.secids)
        sp_prev = P.sess_per_month.reindex(prev.index).fillna(0).values
        lst = pd.DataFrame({s: (prev.index >= fm) if pd.notna(fm) else np.zeros(len(prev), bool)
                            for s, fm in first_m.items()}, index=prev.index)
        num_prev = prev.fillna(0.0).where(lst).sum()
        den_prev = pd.DataFrame(np.repeat(sp_prev[:, None], len(P.secids), 1), index=prev.index,
                                columns=P.secids).where(lst).sum()
        avg_prev = num_prev / den_prev
        cur = vm.iloc[it].reindex(P.secids).fillna(0.0) / P.sess_per_month.get(t, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ab = np.log(cur / avg_prev)
        out["ABVOL"] = ab.where((cur > 0) & (avg_prev > 0) & (lst.sum() >= 6))
    else:
        out["ABVOL"] = np.nan
    capt = P.cap.iloc[it].reindex(P.secids)
    out["SIZE"] = np.log(capt.where(capt > 0))
    return out[CHAR_IDS]


def costs_at(P: Panel, t: pd.Period) -> pd.Series:
    """One-way cost estimate for month t: 0.5 * Abdi–Ranaldo spread + 5 bp (§7)."""
    in_m = np.where(P.month_of == t)[0]
    w1 = slice(in_m[0], in_m[-1] + 1)
    ar, n = _abdi_ranaldo(P.high.iloc[w1].values, P.low.iloc[w1].values, P.close.iloc[w1].values)
    s = pd.Series(np.where(n >= 10, ar, np.nan), index=P.secids)
    return 0.5 * s + 0.0005
