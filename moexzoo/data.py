"""Raw MOEX ISS data -> adjusted daily panels -> monthly panel.

Conventions
-----------
* Prices are raw ISS closes (last trade), stitched across boards by the upstream
  mirror (primary board first, odd-lot board SMAL last). Rows with non-positive
  close or zero volume are treated as no-trade days.
* Splits: a split (D, before=B, after=A) multiplies every price strictly before D
  by c = B/A; volumes are divided by c; dividend amounts with a record date
  strictly before D are multiplied by c (same cascade as the upstream mirror).
* Dividends: sources store the record date (except EX_DATE_SOURCE, which stores the
  ex-date). The ex-date is the k-th own trading day around the record date, k set
  by the settlement regime (T0 / T+2 / T+1). Gross amounts, no tax.
* Daily total return on own trading days: r_k = (P_k + D_k) / P_{k-1} - 1.
  A *calendar* daily return exists only if the share traded on the session and on
  the immediately preceding session; otherwise it is NaN (gap returns still enter
  the total-return index and therefore monthly returns).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Raw loaders
# ----------------------------------------------------------------------------------

def load_tickers() -> dict:
    return json.loads((C.MM / "tickers.json").read_text())


def list_secids() -> list[str]:
    return sorted(p.stem for p in (C.MM / "prices_iss").glob("*.csv"))


def load_prices(secid: str) -> pd.DataFrame:
    df = pd.read_csv(C.MM / "prices_iss" / f"{secid}.csv", parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    return df


def load_splits(secid: str) -> list[tuple[pd.Timestamp, float]]:
    p = C.MM / "splits" / f"{secid}.csv"
    if not p.exists():
        return []
    s = pd.read_csv(p, parse_dates=["date"])
    out = []
    for _, r in s.iterrows():
        b, a = float(r["before"]), float(r["after"])
        if b <= 0 or a <= 0:
            raise ValueError(f"bad split {secid} {r.to_dict()}")
        out.append((pd.Timestamp(r["date"]), b / a))
    return sorted(out)


def split_coef(dates: pd.DatetimeIndex, splits: list[tuple[pd.Timestamp, float]]) -> np.ndarray:
    """Product of B/A over splits whose date is strictly after each date."""
    coef = np.ones(len(dates))
    for d, c in splits:
        coef[np.asarray(dates < d)] *= c
    return coef


def load_dividends(secid: str) -> pd.DataFrame:
    p = C.MM / "dividends" / f"{secid}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["registry_close", "amount", "currency", "source"])
    d = pd.read_csv(p, parse_dates=["registry_close"])
    d["currency"] = d["currency"].fillna("RUB").replace("", "RUB")
    return d


def settlement_lag(source: str, reg: pd.Timestamp) -> int:
    if source == C.EX_DATE_SOURCE:
        return 1
    if reg < pd.Timestamp(C.SETTLEMENT_T2_FROM):
        return 0
    if reg < pd.Timestamp(C.SETTLEMENT_T1_FROM):
        return 2
    return 1


# ----------------------------------------------------------------------------------
# Per-security adjustment
# ----------------------------------------------------------------------------------

@dataclass
class SecurityAdj:
    secid: str
    daily: pd.DataFrame  # own trading days: open/high/low/close adj, volume_adj, value, div, ret, tri
    log: dict = field(default_factory=dict)


def _screen_bad_prints(close_adj: pd.Series, div: pd.Series) -> pd.Index:
    """Single print that jumps by > +100% or < -60% and reverts within +-20% two days later."""
    p = close_adj.values
    d = div.values
    bad = []
    for k in range(1, len(p) - 1):
        r1 = (p[k] + d[k]) / p[k - 1] - 1.0
        if r1 > C.BAD_PRINT_UP or r1 < C.BAD_PRINT_DOWN:
            r2 = (p[k + 1] + d[k + 1]) / p[k] - 1.0
            two = (1 + r1) * (1 + r2) - 1.0
            if abs(two) <= C.BAD_PRINT_REVERSAL_BAND:
                bad.append(close_adj.index[k])
    return pd.Index(bad)


def adjust_security(secid: str, start: str = C.DAILY_START) -> SecurityAdj | None:
    px = load_prices(secid)
    if px.empty:
        return None
    splits = load_splits(secid)
    coef = split_coef(px.index, splits)
    df = pd.DataFrame(index=px.index)
    for col in ("open", "high", "low", "close"):
        df[col] = px[col].astype(float) * coef
    df["volume"] = px["volume"].astype(float) / coef
    df["value"] = px["value"].astype(float)
    df["board"] = px["board"]
    # open/high/low can be missing or zero on thin legacy boards
    for col in ("open", "high", "low"):
        bad = ~(df[col] > 0)
        df.loc[bad, col] = np.nan

    log: dict = {"n_rows": len(df), "splits": len(splits)}

    def attach_dividends(frame: pd.DataFrame) -> pd.Series:
        divs = load_dividends(secid)
        div = pd.Series(0.0, index=frame.index)
        idx = frame.index
        skipped = {"non_rub": 0, "before_first": 0, "after_last": 0, "gap": 0}
        used = 0
        for _, r in divs.iterrows():
            if str(r["currency"]) != "RUB":
                skipped["non_rub"] += 1
                continue
            reg = pd.Timestamp(r["registry_close"])
            last = int(idx.searchsorted(reg, side="right")) - 1
            ex_pos = last - settlement_lag(str(r.get("source", "")), reg) + 1
            if ex_pos <= 0:
                skipped["before_first"] += 1
                continue
            if ex_pos >= len(idx):
                skipped["after_last"] += 1
                continue
            ex = idx[ex_pos]
            rm, em = reg.to_period("M"), ex.to_period("M")
            if not (rm - 1 <= em <= rm + 1):
                skipped["gap"] += 1
                continue
            c = split_coef(pd.DatetimeIndex([reg]), splits)[0]
            div.iloc[ex_pos] += float(r["amount"]) * c
            used += 1
        log["div_used"] = used
        log["div_skipped"] = skipped
        return div

    div = attach_dividends(df)
    bad = _screen_bad_prints(df["close"], div)
    if len(bad):
        df = df.drop(index=bad)
        log["bad_prints"] = [str(x.date()) for x in bad]
        div = attach_dividends(df)  # re-anchor on the cleaned own-trading-day index
    else:
        log["bad_prints"] = []
    df["div"] = div
    prev = df["close"].shift(1)
    df["ret"] = (df["close"] + df["div"]) / prev - 1.0
    df["retx"] = df["close"] / prev - 1.0
    df["tri"] = (1.0 + df["ret"].fillna(0.0)).cumprod()
    df = df[df.index >= pd.Timestamp(start) - pd.Timedelta(days=10)]
    if df.empty:
        return None
    return SecurityAdj(secid, df, log)


# ----------------------------------------------------------------------------------
# Market, calendar, risk-free
# ----------------------------------------------------------------------------------

def load_mcftrr() -> pd.Series:
    m = pd.read_csv(C.MM / "indices" / "MCFTRR.csv", parse_dates=["date"]).set_index("date")["close"]
    return m.sort_index().astype(float)


def load_key_rate() -> pd.Series:
    k = pd.read_csv(C.RAW / "moexutils" / "key_rate.csv", parse_dates=["date"]).set_index("date")["rate"]
    return k.sort_index().astype(float)


def monthly_rf(months: pd.PeriodIndex) -> pd.Series:
    """Monthly risk-free proxy: CBR key (pre-2013: refinancing) rate in force on the
    first day of the month, compounded to a monthly rate."""
    k = load_key_rate()
    out = {}
    for m in months:
        d = m.start_time
        r = k[k.index <= d]
        out[m] = (1.0 + r.iloc[-1] / 100.0) ** (1.0 / 12.0) - 1.0 if len(r) else np.nan
    return pd.Series(out)


# ----------------------------------------------------------------------------------
# Shares outstanding (for size, turnover, value weights)
# ----------------------------------------------------------------------------------

def _parse_date_cell(x) -> pd.Timestamp | None:
    if isinstance(x, (pd.Timestamp,)) or hasattr(x, "year"):
        return pd.Timestamp(x)
    if isinstance(x, str):
        m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", x)
        if m:
            return pd.Timestamp(f"{m.group(3)}-{m.group(2)}-{m.group(1)}")
    return None


def load_index_bases() -> pd.DataFrame:
    """Rows: (effective_from, secid, shares_raw) from MOEX index calculation bases."""
    x = pd.ExcelFile(C.RAW / "moexutils" / "stock-index-base.xlsx")
    rows = []
    for s in x.sheet_names:
        if s == "help":
            continue
        d = pd.read_excel(x, s, header=None)
        eff = _parse_date_cell(d.iloc[1, 2]) or _parse_date_cell(s)
        hdr = None
        for i in range(min(10, len(d))):
            vals = [str(v) for v in d.iloc[i].tolist()]
            if "Code" in vals and any("Number of issued shares" in v for v in vals):
                hdr = i
                break
        if hdr is None:
            raise ValueError(f"no header in sheet {s}")
        cols = [str(v) for v in d.iloc[hdr].tolist()]
        ci = cols.index("Code")
        si = [j for j, v in enumerate(cols) if "Number of issued shares" in v][0]
        for i in range(hdr + 1, len(d)):
            code = d.iloc[i, ci]
            if isinstance(code, str) and "Исключены" in code:
                break  # excluded names listed below carry no share counts
            sh = d.iloc[i, si]
            if isinstance(code, str) and code.strip() and pd.notna(sh):
                try:
                    shv = float(sh)
                except (TypeError, ValueError):
                    continue
                if shv > 0:
                    rows.append((eff, code.strip(), shv))
    out = pd.DataFrame(rows, columns=["effective_from", "secid", "shares_raw"])
    return out.sort_values(["secid", "effective_from"]).reset_index(drop=True)


def load_current_issue_amounts() -> pd.Series:
    d = pd.read_csv(C.RAW / "foykes" / "moex_stocks.csv")
    d = d[["TRADE_CODE", "ISSUE_AMOUNT"]].dropna()
    d["ISSUE_AMOUNT"] = pd.to_numeric(d["ISSUE_AMOUNT"].astype(str).str.replace(",", "."), errors="coerce")
    d = d.dropna().drop_duplicates("TRADE_CODE")
    return d.set_index("TRADE_CODE")["ISSUE_AMOUNT"]


ISSUE_AMOUNT_DATE = pd.Timestamp("2026-05-11")
