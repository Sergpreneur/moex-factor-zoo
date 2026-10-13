"""Exploratory (second revision): the 2013 board switch in the price source.

From 25 March 2013 the source takes the daily rows of some shares from the T+2 board (TQBR), which until the switch of
the main board on 2 September 2013 carried a small fraction of their trading volume. Volume-based characteristics of
these shares are therefore understated in March-August 2013, and comparisons with that window (abnormal volume, 12-month
liquidity measures) are distorted until August 2014. This script sets the volume- and trading-based characteristics of
every share with a T+ board row in that window to missing for formation months 2013-03 to 2014-08, recomputes those
long-short portfolios and reruns the inference ladder with them in place of the baseline series.

Output: results/summary_board_volume.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import chars as CH  # noqa: E402
from moexzoo import config as C  # noqa: E402
from moexzoo import data as D  # noqa: E402
from moexzoo import portfolios as PF  # noqa: E402

_spec = importlib.util.spec_from_file_location("inf04", Path(__file__).resolve().parent / "04_inference.py")
inf04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf04)

WINDOW = ("2013-03-25", "2013-09-01")          # T+ board rows before the main board moved to T+2
MASK_MONTHS = ("2013-03", "2014-08")           # formation months whose characteristics use that window
VOLCHARS = ["ABVOL", "TURN", "DVOL", "AMIHUD", "CVVAL", "ZERO", "SPREAD"]


def sample(s: pd.Series) -> pd.Series:
    return s[(s.index >= pd.Period(C.SAMPLE_START, "M")) & (s.index <= pd.Period(C.SAMPLE_END, "M"))]


def affected_secids() -> dict:
    out = {}
    for s in D.list_secids():
        px = pd.read_csv(C.MM / "prices_iss" / f"{s}.csv", usecols=["date", "board"], parse_dates=["date"])
        w = px[(px["date"] >= WINDOW[0]) & (px["date"] <= WINDOW[1])]
        n = int(w["board"].astype(str).str.startswith("TQ").sum())
        if n:
            out[s] = n
    return out


def value_ratio(secid: str) -> float:
    """Mean daily trading value in April-August 2013 relative to 9 January - 22 March 2013, in %."""
    px = pd.read_csv(C.MM / "prices_iss" / f"{secid}.csv", usecols=["date", "value"], parse_dates=["date"])
    px = px.set_index("date")["value"]
    return float(100 * px.loc["2013-04-01":"2013-08-30"].mean() / px.loc["2013-01-09":"2013-03-22"].mean())


def main() -> None:
    assert not os.environ.get("MOEXZOO_VARIANT"), "run without MOEXZOO_VARIANT"
    aff = affected_secids()
    ratios = {s: value_ratio(s) for s in ("SBER", "LKOH", "VTBR", "GAZP", "GMKN")}
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    base = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")
    base.index = pd.PeriodIndex(base.index, freq="M")
    base = base[CH.CHAR_IDS]
    # sanity: the unmasked recomputation reproduces the baseline series
    chk = sample(PF.long_short(panel, "ABVOL", CH.SIGN["ABVOL"]).ret)
    assert np.allclose(chk.values, base["ABVOL"].reindex(chk.index).values, equal_nan=True)
    m0, m1 = pd.Period(MASK_MONTHS[0], "M"), pd.Period(MASK_MONTHS[1], "M")
    mask = panel["secid"].isin(aff) & (panel["month"] >= m0) & (panel["month"] <= m1)
    pm = panel.copy()
    pm.loc[mask, VOLCHARS] = np.nan
    F = base.copy()
    for k in VOLCHARS:
        F[k] = sample(PF.long_short(pm, k, CH.SIGN[k]).ret).reindex(F.index)
    lad0, info0 = inf04.ladder(base)
    lad, info = inf04.ladder(F)
    surv0 = [k for k in CH.CHAR_IDS if lad0.loc[k, "R6_rw"]]
    surv = [k for k in CH.CHAR_IDS if lad.loc[k, "R6_rw"]]
    out = {
        "n_secids": len(aff), "secids": aff, "value_ratio_pct": ratios,
        "n_u100_masked": int((mask & panel["U100"]).sum()),
        "masked_chars": VOLCHARS, "mask_months": MASK_MONTHS,
        "baseline": {"survivors": surv0, "crit": info0["rw_crit_first"], "counts": info0["counts"]},
        "masked": {"survivors": surv, "crit": info["rw_crit_first"], "counts": info["counts"],
                   "t": {k: float(lad.loc[k, "t"]) for k in CH.CHAR_IDS},
                   "p_rw": {k: float(lad.loc[k, "p_rw"]) for k in CH.CHAR_IDS if "p_rw" in lad.columns}},
        "t_base": {k: float(lad0.loc[k, "t"]) for k in VOLCHARS},
        "t_masked": {k: float(lad.loc[k, "t"]) for k in VOLCHARS},
        "max_abs_dt": float(np.nanmax(np.abs(lad["t"].values - lad0["t"].values))),
    }
    print("affected", len(aff), "U100 stock-months masked", out["n_u100_masked"])
    print("survivors", surv0, "->", surv, "crit %.3f -> %.3f" % (info0["rw_crit_first"], info["rw_crit_first"]))
    for k in VOLCHARS:
        print(f"  {k:7s} t {out['t_base'][k]:6.2f} -> {out['t_masked'][k]:6.2f}")
    (C.RESULTS / "summary_board_volume.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
