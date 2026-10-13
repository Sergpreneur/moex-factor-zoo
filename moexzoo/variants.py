"""Exploratory data-construction sensitivities (second revision; prereg/deviations.md, items 24 and 27).

Each variant rebuilds the panels from the raw data in its own folders (data/interim_<v>, data/processed_<v>,
results/variants/<v>), so the pre-specified outputs are never overwritten. The frozen modules moexzoo/config.py
and moexzoo/data.py are not modified (their SHA-256 hashes still match prereg/FREEZE.txt): scripts/run_variant.py
calls activate(), which redirects the three folders in config and replaces data.adjust_security by
adjust_security() below, and then runs the requested pipeline script.

  divcal        corrected dividend calendar: the ex-date is located on the exchange session calendar (MCFTRR
                dates and the share's own trading days); before 2 September 2013 the settlement regime is read
                from the board of the price (T+2 for the T+ boards TQ*, T0 for the legacy boards); the payment
                is credited on the share's first own trading day on or after the ex-date
  divcal_dedup  as divcal, and RUB payments of a share that share a record date are treated as possible
                duplicates (only the larger is kept)
  yahoo_early   as the baseline, but every date of the ex-date source (config.EX_DATE_SOURCE) is moved two
                exchange sessions earlier
  yahoo_late    as yahoo_early, two sessions later
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

VARIANTS = {
    "divcal": dict(rule="corrected", dedup=False, ex_shift=0),
    "divcal_dedup": dict(rule="corrected", dedup=True, ex_shift=0),
    "yahoo_early": dict(rule="plan", dedup=False, ex_shift=-2),
    "yahoo_late": dict(rule="plan", dedup=False, ex_shift=2),
}

ACTIVE: str = ""
_SESSIONS: pd.DatetimeIndex | None = None


def activate(name: str) -> None:
    """Select a variant: redirect the interim, processed and results folders and swap in the variant
    adjustment. An empty name leaves everything as pre-specified."""
    global ACTIVE
    if not name:
        return
    if name not in VARIANTS:
        raise ValueError(f"unknown MOEXZOO_VARIANT {name!r}; known: {', '.join(VARIANTS)}")
    ACTIVE = name
    C.INTERIM = C.ROOT / "data" / f"interim_{name}"
    C.PROCESSED = C.ROOT / "data" / f"processed_{name}"
    C.RESULTS = C.ROOT / "results" / "variants" / name
    D.adjust_security = adjust_security


def exchange_sessions() -> pd.DatetimeIndex:
    """Exchange session calendar: MCFTRR publication dates (cached)."""
    global _SESSIONS
    if _SESSIONS is None:
        _SESSIONS = pd.DatetimeIndex(sorted(set(D.load_mcftrr().index)))
    return _SESSIONS


def ex_position_corrected(idx: pd.DatetimeIndex, boards: pd.Series, reg: pd.Timestamp, source: str) -> int:
    """Corrected ex-date rule (divcal): the ex-date is located on the exchange session calendar (MCFTRR dates
    and the share's own trading days) and the payout is credited on the share's first own trading day on or
    after it. Before 2 September 2013 the settlement regime is read from the board of the price: T+2 for the
    T+ boards (TQ*), T0 for the legacy boards. Returns the position in idx (may be len(idx) if the share never
    trades again)."""
    cal = exchange_sessions().union(idx)
    if source == C.EX_DATE_SOURCE:
        ex = cal[min(int(cal.searchsorted(reg, side="left")), len(cal) - 1)]
        return int(idx.searchsorted(ex, side="left"))
    j = int(cal.searchsorted(reg, side="right")) - 1  # last session on or before the record date
    if reg < pd.Timestamp(C.SETTLEMENT_T2_FROM):
        k_own = int(idx.searchsorted(reg, side="right")) - 1
        b = str(boards.iloc[max(k_own, 0)]) if len(boards) else ""
        k = 2 if b.startswith("TQ") else 0
    elif reg < pd.Timestamp(C.SETTLEMENT_T1_FROM):
        k = 2
    else:
        k = 1
    jj = {0: j + 1, 2: j - 1, 1: j}[k]
    jj = min(max(jj, 0), len(cal) - 1)
    return int(idx.searchsorted(cal[jj], side="left"))


def shift_sessions(d: pd.Timestamp, n: int, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Move a date n exchange sessions (MCFTRR dates and the share's own trading days) from the first session on
    or after it."""
    cal = exchange_sessions().union(idx)
    i = min(int(cal.searchsorted(d, side="left")), len(cal) - 1)
    return cal[min(max(i + n, 0), len(cal) - 1)]


def adjust_security(secid: str, start: str = C.DAILY_START) -> D.SecurityAdj | None:
    """data.adjust_security with the dividend rule of the active variant; identical otherwise."""
    spec = VARIANTS[ACTIVE]
    px = D.load_prices(secid)
    if px.empty:
        return None
    splits = D.load_splits(secid)
    coef = D.split_coef(px.index, splits)
    df = pd.DataFrame(index=px.index)
    for col in ("open", "high", "low", "close"):
        df[col] = px[col].astype(float) * coef
    df["volume"] = px["volume"].astype(float) / coef
    df["value"] = px["value"].astype(float)
    df["board"] = px["board"]
    for col in ("open", "high", "low"):
        bad = ~(df[col] > 0)
        df.loc[bad, col] = np.nan

    log: dict = {"n_rows": len(df), "splits": len(splits), "variant": ACTIVE}

    def attach_dividends(frame: pd.DataFrame) -> pd.Series:
        divs = D.load_dividends(secid)
        div = pd.Series(0.0, index=frame.index)
        idx = frame.index
        skipped = {"non_rub": 0, "before_first": 0, "after_last": 0, "gap": 0}
        used = moved = 0
        if spec["dedup"] and len(divs):
            rub = divs[divs["currency"].astype(str) == "RUB"].copy()
            rub["amount"] = rub["amount"].astype(float)
            keep = rub.sort_values("amount").groupby("registry_close").tail(1).index
            log["div_dedup_dropped"] = int(len(rub) - len(keep))
            divs = pd.concat([rub.loc[keep], divs[divs["currency"].astype(str) != "RUB"]])
        for _, r in divs.iterrows():
            if str(r["currency"]) != "RUB":
                skipped["non_rub"] += 1
                continue
            source = str(r.get("source", ""))
            reg0 = pd.Timestamp(r["registry_close"])
            last0 = int(idx.searchsorted(reg0, side="right")) - 1
            ex_plan = last0 - D.settlement_lag(source, reg0) + 1
            reg = reg0
            if spec["ex_shift"] and source == C.EX_DATE_SOURCE:
                reg = shift_sessions(reg0, spec["ex_shift"], idx)
            if spec["rule"] == "corrected":
                ex_pos = ex_position_corrected(idx, frame["board"], reg, source)
            else:
                last = int(idx.searchsorted(reg, side="right")) - 1
                ex_pos = last - D.settlement_lag(source, reg) + 1
            moved += int(ex_pos != ex_plan)
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
            c = D.split_coef(pd.DatetimeIndex([reg0]), splits)[0]
            div.iloc[ex_pos] += float(r["amount"]) * c
            used += 1
        log["div_used"] = used
        log["div_skipped"] = skipped
        log["div_moved"] = moved
        return div

    div = attach_dividends(df)
    bad = D._screen_bad_prints(df["close"], div)
    if len(bad):
        df = df.drop(index=bad)
        log["bad_prints"] = [str(x.date()) for x in bad]
        div = attach_dividends(df)
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
    return D.SecurityAdj(secid, df, log)
