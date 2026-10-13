"""Build adjusted daily panels and the monthly panel from the raw mirror.

Outputs (data/interim/):
  daily_{close,open,high,low,value,volume,ret,retx,div,traded}.parquet  (session x secid)
  market_daily.parquet                     session-level MCFTRR level and return
  monthly.parquet                          long panel (month, secid) with returns, liquidity, cap
  build_log.json                           per-security adjustment log (splits, dividends, screens)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402
from moexzoo import data as D  # noqa: E402


def main() -> None:
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    tick = D.load_tickers()
    secids = D.list_secids()
    adj: dict[str, pd.DataFrame] = {}
    build_log: dict[str, dict] = {}
    for s in secids:
        a = D.adjust_security(s)
        if a is None:
            continue
        adj[s] = a.daily
        build_log[s] = a.log
    print(f"securities with data: {len(adj)} of {len(secids)}")

    # ---- session calendar ---------------------------------------------------------
    mk = D.load_mcftrr()
    counts = pd.Series(0, dtype=int)
    allidx = pd.concat([pd.Series(1, index=df.index) for df in adj.values()])
    counts = allidx.groupby(level=0).sum()
    sessions = set(mk.index) | set(counts[counts >= C.MIN_STOCKS_FOR_CALENDAR_DAY].index)
    cal = pd.DatetimeIndex(sorted(d for d in sessions if d >= pd.Timestamp(C.DAILY_START)))
    print(f"sessions: {len(cal)} from {cal[0].date()} to {cal[-1].date()}")

    # ---- wide daily panels -------------------------------------------------------
    cols = {k: {} for k in ("close", "open", "high", "low", "value", "volume", "ret", "retx", "div", "tri")}
    pos = pd.Series(np.arange(len(cal)), index=cal)
    for s, df in adj.items():
        df = df[df.index.isin(cal)]
        if df.empty:
            continue
        p = pos.reindex(df.index).values
        # a calendar daily return needs the previous session traded AND a gap of at
        # most MAX_SESSION_GAP_DAYS calendar days (drops the 2022-02-25 -> 2022-03-24
        # closure, keeps New Year breaks)
        cal_gap = np.r_[np.inf, np.diff(df.index.values).astype("timedelta64[D]").astype(float)]
        consecutive = np.r_[False, np.diff(p) == 1] & (cal_gap <= C.MAX_SESSION_GAP_DAYS)
        ret = df["ret"].where(consecutive)
        retx = df["retx"].where(consecutive)
        for k in ("close", "open", "high", "low", "value", "volume", "div", "tri"):
            cols[k][s] = df[k]
        cols["ret"][s] = ret
        cols["retx"][s] = retx
    for k, dct in cols.items():
        w = pd.DataFrame(dct).reindex(cal)
        w.columns.name = "secid"
        w.to_parquet(C.INTERIM / f"daily_{k}.parquet")
        print(k, w.shape)
    traded = pd.DataFrame(cols["close"]).reindex(cal).notna()

    # ---- market -------------------------------------------------------------------
    mkt = mk.reindex(cal)
    # daily market return against the previous actual MCFTRR print; NaN when the gap between the two prints
    # exceeds MAX_SESSION_GAP_DAYS (the index has no print on 2022-03-24, so the return on 2022-03-25 would
    # otherwise span the whole trading suspension)
    mk_s = mk[mk.index >= cal[0]]
    r_act = mk_s / mk_s.shift(1) - 1.0
    gap_act = mk_s.index.to_series().diff().dt.days
    r_act[gap_act > C.MAX_SESSION_GAP_DAYS] = np.nan
    mret = r_act.reindex(cal)
    pd.DataFrame({"level": mkt, "ret": mret}).to_parquet(C.INTERIM / "market_daily.parquet")

    # ---- monthly panel -------------------------------------------------------------
    close = pd.DataFrame(cols["close"]).reindex(cal)
    tri = pd.DataFrame(cols["tri"]).reindex(cal)
    value = pd.DataFrame(cols["value"]).reindex(cal).fillna(0.0)
    months = cal.to_period("M")
    last_session = pd.Series(cal, index=cal).groupby(months).max()
    month_index = pd.PeriodIndex(last_session.index, freq="M")

    # last trade within each month (per security)
    tri_m = tri.groupby(months).last()  # last non-NaN in month
    close_m = close.groupby(months).last()
    ndays = traded.groupby(months).sum()
    value_m = value.groupby(months).sum()
    # last trade date in month
    trade_dates = traded.apply(lambda col: pd.Series(np.where(col.values, cal.values, np.datetime64("NaT")), index=cal))
    last_trade = trade_dates.groupby(months).max()
    # sessions remaining after last trade within the month
    sess_pos = pd.Series(np.arange(len(cal)), index=cal)
    last_sess_pos = sess_pos.groupby(months).max()

    traded_m = ndays > 0
    prev_traded = traded_m.shift(1, fill_value=False)
    ret_m = (tri_m / tri_m.shift(1) - 1.0).where(traded_m & prev_traded)
    retx_m = (close_m / close_m.shift(1) - 1.0).where(traded_m & prev_traded)

    # sessions between last trade and month end
    lt_pos = last_trade.apply(lambda col: col.map(lambda d: sess_pos.get(d, np.nan) if pd.notna(d) else np.nan))
    gap_to_end = (-(lt_pos.sub(last_sess_pos.values, axis=0)))

    mret_m = (mk.groupby(mk.index.to_period("M")).last())
    mret_m = (mret_m / mret_m.shift(1) - 1.0).reindex(month_index)
    rf = D.monthly_rf(month_index)

    long = pd.DataFrame({
        "ret": ret_m.stack(),
        "retx": retx_m.stack(),
        "close": close_m.stack(),
        "ndays": ndays.stack(),
        "value_m": value_m.stack(),
        "gap_to_end": gap_to_end.stack(),
    })
    long.index.names = ["month", "secid"]
    long = long[(long["ndays"] > 0) | long["ret"].notna()]
    long = long.reset_index()

    # ---- shares outstanding and market cap ---------------------------------------
    bases = D.load_index_bases()
    issue = D.load_current_issue_amounts()
    month_end = last_session.copy()
    month_end.index = month_index
    long["month_end"] = long["month"].map(month_end)
    shares = np.full(len(long), np.nan)
    src = np.array([""] * len(long), dtype=object)
    base_by = {s: g for s, g in bases.groupby("secid")}
    splits_by = {s: D.load_splits(s) for s in long["secid"].unique()}
    for s, g in long.groupby("secid"):
        idx = g.index
        dates = pd.DatetimeIndex(g["month_end"].values)
        sp = splits_by.get(s, [])
        coef = D.split_coef(dates, sp)  # adjusted price = raw * coef
        if s in base_by:
            b = base_by[s].sort_values("effective_from")
            eff = pd.DatetimeIndex(b["effective_from"].values)
            sh = b["shares_raw"].values
            # point-in-time: latest base effective on or before the month end
            k = eff.searchsorted(dates, side="right") - 1
            pit = k >= 0
            raw_counts = np.where(pit, sh[np.clip(k, 0, len(sh) - 1)], sh[0])
            # shares on the adjusted (current) scale: raw shares / coef(base effective date)
            eff_coef = D.split_coef(eff, sp)
            adj_counts = np.where(pit, raw_counts / eff_coef[np.clip(k, 0, len(sh) - 1)], sh[0] / eff_coef[0])
            shares[idx] = adj_counts
            src[idx] = np.where(pit, "index_base_pit", "index_base_backfill")
        elif s in issue.index:
            # current number of shares (11.05.2026) is already on the post-split scale
            shares[idx] = float(issue[s])
            src[idx] = "issue_amount_current"
    long["shares_adj"] = shares
    long["shares_src"] = src
    long["cap"] = long["close"] * long["shares_adj"]  # adjusted close x adjusted shares = raw cap
    long["mkt_ret"] = long["month"].map(mret_m)
    long["rf"] = long["month"].map(rf)
    long["type"] = long["secid"].map(lambda s: tick.get(s, {}).get("type", ""))
    long.to_parquet(C.INTERIM / "monthly.parquet")

    mkt_m = pd.DataFrame({"mkt_ret": mret_m, "rf": rf})
    mkt_m.index = mkt_m.index.astype(str)
    mkt_m.to_parquet(C.INTERIM / "market_monthly.parquet")
    (C.INTERIM / "build_log.json").write_text(json.dumps(build_log, ensure_ascii=False, indent=0, default=str))
    print("monthly rows", len(long), "months", long["month"].nunique(), "secids", long["secid"].nunique())
    print(long["shares_src"].value_counts())


if __name__ == "__main__":
    main()
