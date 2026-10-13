"""Helpers for the exploratory analyses added in the revision (prereg/deviations.md, section D).

Nothing here is used by the pre-specified pipeline (scripts 01–05d). The extended long–short function
reproduces moexzoo.portfolios.long_short exactly for its default arguments (checked in 05e_revision.py)
and adds per-leg returns and costs, a benchmark return, a short-leg eligibility filter, a short-sale ban,
and a square-root price-impact charge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .portfolios import _groups


# ----------------------------------------------------------------------------------
# masked OLS with several regressors (columns of y are separate regressions)
# ----------------------------------------------------------------------------------

def ols_multi_masked(y: np.ndarray, X: np.ndarray):
    """y: (T, N) with NaN; X: (T, p) with NaN (common regressors). Regress each column of y on [1, X]
    using the rows where y and all regressors are observed. Returns (coef (N, p+1), resid (T, N), n (N))."""
    T, N = y.shape
    p = X.shape[1]
    xok = ~np.isnan(X).any(1)
    mask = ~np.isnan(y) & xok[:, None]
    n = mask.sum(0).astype(float)
    Z = np.column_stack([np.ones(T), np.nan_to_num(X)])  # (T, p+1)
    w = mask.astype(float)
    Y = np.where(mask, y, 0.0)
    # normal equations per column: S_n = sum_t w_tn z_t z_t', b_n = sum_t w_tn z_t y_tn
    S = np.einsum("tn,ti,tj->nij", w, Z, Z)
    b = np.einsum("tn,ti->ni", Y, Z)
    coef = np.full((N, p + 1), np.nan)
    ok = n >= p + 2
    if ok.any():
        det = np.linalg.det(S[ok])
        good = np.abs(det) > 1e-18
        sol = np.full((ok.sum(), p + 1), np.nan)
        sol[good] = np.linalg.solve(S[ok][good], b[ok][good][..., None])[..., 0]
        coef[ok] = sol
    fitted = Z @ coef.T  # (T, N)
    resid = np.where(mask, y - fitted, np.nan)
    return coef, resid, n


# ----------------------------------------------------------------------------------
# a size factor (SMB) from the sample itself
# ----------------------------------------------------------------------------------

def build_smb(P, U: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Value-weighted small-minus-big returns. At the end of each month s-1, the shares in UALL with a market
    capitalisation are split at the median capitalisation; within each half, weights are proportional to
    capitalisation at s-1 and held fixed through month s. Returns (monthly SMB indexed by holding month,
    daily SMB indexed by session). Missing returns of held names are excluded (weights renormalised)."""
    U = U.copy()
    U["month"] = pd.PeriodIndex(U["month"], freq="M")
    ua = U[U["UALL"]]
    ret_d = P.ret
    month_of = P.month_of
    smb_m, smb_d = {}, {}
    months = sorted(ua["month"].unique())
    for f in months:
        h = f + 1
        if f not in P.cap.index or h not in P.R.index:
            continue
        names = ua.loc[ua["month"] == f, "secid"].values
        cap = P.cap.loc[f].reindex(names)
        cap = cap[cap > 0].dropna()
        if len(cap) < 20:
            continue
        med = cap.median()
        small, big = cap[cap <= med], cap[cap > med]
        ws, wb = small / small.sum(), big / big.sum()
        Rh = P.R.loc[h]
        # monthly: missing -> excluded, renormalise
        def wavg(w, r):
            r = r.reindex(w.index)
            ok = r.notna()
            return float((w[ok] * r[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        smb_m[h] = wavg(ws, Rh) - wavg(wb, Rh)
        sel = np.where(month_of == h)[0]
        if len(sel) == 0:
            continue
        rd = ret_d.iloc[sel]
        for wv, sgn in ((ws, 1.0), (wb, -1.0)):
            r = rd.reindex(columns=wv.index)
            okm = r.notna().values
            num = np.nansum(r.values * wv.values[None, :], 1)
            den = (okm * wv.values[None, :]).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                part = np.where(den > 0, num / den, np.nan)
            for d, v in zip(rd.index, part):
                smb_d[d] = smb_d.get(d, 0.0) + sgn * v
    sm = pd.Series(smb_m).sort_index()
    sm.index = pd.PeriodIndex(sm.index, freq="M")
    sd = pd.Series(smb_d).sort_index()
    sd = sd.reindex(P.sessions)
    # the 14-day gap rule applies to the factor as to the market
    gap = pd.Series(P.sessions, index=P.sessions).diff().dt.days
    sd[gap > C.MAX_SESSION_GAP_DAYS] = np.nan
    sd[P.mret.isna()] = np.nan
    return sm, sd


def chars_2f(P, t: pd.Period, smb_m: pd.Series, smb_d: pd.Series) -> pd.DataFrame:
    """RESMOM and IVOL with residuals from a two-factor model (market and SMB) instead of the CAPM;
    windows, minimum data and scaling exactly as in moexzoo.chars."""
    out = pd.DataFrame(index=P.secids, dtype=float)
    R = P.R
    it = R.index.get_loc(t)
    if it - 35 >= 0:
        blk = R.iloc[it - 35: it]
        mk = (P.mkt_m - P.rf_m).reindex(blk.index).values
        sm = smb_m.reindex(blk.index).values
        y = blk.sub(P.rf_m.reindex(blk.index).values, axis=0).values
        coef, resid, n = ols_multi_masked(y, np.column_stack([mk, sm]))
        last11 = resid[-11:]
        full11 = (~np.isnan(last11)).sum(0) == 11
        with np.errstate(invalid="ignore", divide="ignore"):
            rm = np.nansum(last11, 0) / np.nanstd(last11, 0, ddof=1)
        out["RESMOM2F"] = pd.Series(np.where((n >= 24) & full11, rm, np.nan), index=R.columns).reindex(P.secids)
    else:
        out["RESMOM2F"] = np.nan
    in_m = np.where(P.month_of == t)[0]
    w1 = slice(in_m[0], in_m[-1] + 1)
    r1 = P.ret.iloc[w1].values
    X1 = np.column_stack([P.mret.iloc[w1].values, smb_d.iloc[w1].values])
    coef, e1, ne1 = ols_multi_masked(r1, X1)
    with np.errstate(invalid="ignore"):
        out["IVOL2F"] = pd.Series(np.where(ne1 >= C.MIN_OBS_1M, np.nanstd(e1, 0, ddof=1), np.nan), index=P.secids)
    return out


# ----------------------------------------------------------------------------------
# extended long–short portfolios (equal weights)
# ----------------------------------------------------------------------------------

def ls_ext(panel: pd.DataFrame, char: str, sign: int, *, universe: str = "U100", ret_col: str = "R_next",
           ret_col_long: str | None = None, ret_col_short: str | None = None, n_groups: int = C.N_GROUPS,
           min_n: int = C.MIN_STOCKS_PER_SORT, short_ok_col: str | None = None, ban_months: tuple = (),
           ban_mode: str | None = None, aum: float | None = None, impact_y: float = 1.0,
           count_col: str | None = None, vol_col: str = "TVOL") -> pd.DataFrame:
    """Equal-weighted quintile long–short with per-leg detail. With default arguments `ret`, `turnover`
    and `cost` equal those of portfolios.long_short (EW). Options:
      ret_col_long / ret_col_short: holding-return columns for each leg (default ret_col);
      short_ok_col: boolean column; the short leg holds only eligible names of the short quintile, and if
        none is eligible, all eligible names of the sorted universe (flag `fallback`);
      ban_months, ban_mode: in these holding months the short leg is replaced by cash ('longonly') or the
        whole position by cash ('flat');
      aum: notional of each leg in roubles; adds a square-root impact charge
        impact_y * sigma_d * sqrt(|dw| * aum / ADV) per unit traded (sigma_d: daily volatility in column
        `vol_col`, ADV: mean daily trading value of month t in column `adv`; cross-sectional medians if missing);
      count_col: boolean column whose True values are counted in each leg (columns `flag_l`, `flag_s`).
    Returns a DataFrame indexed by holding month."""
    rcl = ret_col_long or ret_col
    rcs = ret_col_short or ret_col
    payers_only = char == "DY"
    rows = {}
    prev = None
    for t, g in panel.groupby("month", sort=True):
        g = g[g[universe]].set_index("secid")
        x = g[char]
        if payers_only:
            x = x.where(x > 0)
        x = x.dropna()
        hm = t + 1
        if len(x) < min_n:
            rows[hm] = dict(ret=np.nan)
            prev = None
            continue
        grp = _groups(x, n_groups)
        hi, lo = grp.index[grp == n_groups], grp.index[grp == 1]
        if sign < 0:
            hi, lo = lo, hi
        fallback = False
        if short_ok_col is not None:
            ok = g[short_ok_col].fillna(False).astype(bool)
            lo_ok = lo[ok.reindex(lo).values]
            if len(lo_ok) == 0:
                lo_ok = x.index[ok.reindex(x.index).values]
                fallback = True
            lo = lo_ok
        banned = hm in ban_months
        wl = pd.Series(1.0 / len(hi), index=hi)
        ws = pd.Series(1.0 / len(lo), index=lo) if len(lo) else pd.Series(dtype=float)
        if banned and ban_mode == "longonly":
            ws = pd.Series(dtype=float)
        if banned and ban_mode == "flat":
            wl = pd.Series(dtype=float)
            ws = pd.Series(dtype=float)
        rL = g[rcl].reindex(wl.index).fillna(0.0)
        rS = g[rcs].reindex(ws.index).fillna(0.0)
        rl = float((wl * rL).sum()) if len(wl) else 0.0
        rs = float((ws * rS).sum()) if len(ws) else 0.0
        bench = float(g[ret_col].reindex(x.index).fillna(0.0).mean())
        if prev is None:
            prev_l, prev_s = pd.Series(dtype=float), pd.Series(dtype=float)
        else:
            prev_l, prev_s = prev
        cst = g["cost"]
        cm = cst.median()
        fill = cm if pd.notna(cm) else 0.005

        def leg_cost(w_new, w_old):
            idx = w_new.index.union(w_old.index)
            dw = (w_new.reindex(idx, fill_value=0.0) - w_old.reindex(idx, fill_value=0.0)).abs()
            c = cst.reindex(idx).fillna(fill)
            imp = 0.0
            if aum is not None:
                sig = g[vol_col].reindex(idx)
                sig = sig.fillna(g[vol_col].median())
                adv = g["adv"].reindex(idx)
                adv = adv.where(adv > 0).fillna(g["adv"].median())
                imp = float((dw * impact_y * sig * np.sqrt(dw * aum / adv)).sum())
            return float(dw.sum()), float((dw * c).sum()), imp

        to_l, c_l, i_l = leg_cost(wl, prev_l)
        to_s, c_s, i_s = leg_cost(ws, prev_s)
        fl = fs = 0
        if count_col is not None:
            fl = int(g[count_col].reindex(wl.index).fillna(False).astype(bool).sum())
            fs = int(g[count_col].reindex(ws.index).fillna(False).astype(bool).sum())
        rows[hm] = dict(ret=rl - rs, long=rl, short=rs, bench=bench, to_l=to_l, to_s=to_s, cost_l=c_l, cost_s=c_s,
                        imp_l=i_l, imp_s=i_s, n_l=len(wl), n_s=len(ws), fallback=fallback, flag_l=fl, flag_s=fs)
        # drift with the baseline return column (as in portfolios.long_short)
        dl = wl * (1 + g[ret_col].reindex(wl.index).fillna(0.0)) if len(wl) else wl
        ds = ws * (1 + g[ret_col].reindex(ws.index).fillna(0.0)) if len(ws) else ws
        prev = (dl / dl.sum() if len(dl) and dl.sum() > 0 else wl, ds / ds.sum() if len(ds) and ds.sum() > 0 else ws)
    df = pd.DataFrame(rows).T
    df.index = pd.PeriodIndex(df.index, freq="M")
    df = df.sort_index()
    for c in df.columns:
        if c != "fallback":
            df[c] = df[c].astype(float)
    df["turnover"] = df["to_l"] + df["to_s"]
    df["cost"] = df["cost_l"] + df["cost_s"]
    df["impact"] = df["imp_l"] + df["imp_s"]
    return df
