"""Long–short factor returns for the baseline and every robustness variant of the plan (§5, §7).

Outputs results/factors/<variant>.parquet (holding month x 25 factors) and
results/factors/diagnostics_baseline.parquet (names per leg, turnover, costs, cash names).
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
from moexzoo import portfolios as PF  # noqa: E402

OUT = C.RESULTS / "factors"

import os  # noqa: E402

VARIANTS = {
    "baseline": dict(),
    "vw": dict(weighting="VW"),
    "u50": dict(universe="U50"),
    "uall": dict(universe="UALL"),
    "terciles": dict(n_groups=3),
    "lag1": dict(ret_col="Rlag_next"),
    "u100ord": dict(universe="U100ORD"),
    "priceonly": dict(ret_col="Rx_next"),
}


def sample(s: pd.Series) -> pd.Series:
    return s[(s.index >= pd.Period(C.SAMPLE_START, "M")) & (s.index <= pd.Period(C.SAMPLE_END, "M"))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MOEXZOO_FACTORS_BASELINE_ONLY"):
        # data-construction sensitivities (moexzoo/variants.py) need only the baseline factors
        for k in [v for v in VARIANTS if v != "baseline"]:
            VARIANTS.pop(k)
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    t0 = time.time()
    diag_rows = []
    for vname, kw in VARIANTS.items():
        cols = {}
        net, net2 = {}, {}
        for k in CH.CHAR_IDS:
            res = PF.long_short(panel, k, CH.SIGN[k], **kw)
            cols[k] = sample(res.ret)
            if vname == "baseline":
                net[k] = sample(res.ret - res.cost)
                res2 = PF.long_short(panel, k, CH.SIGN[k], cost_mult=2.0)
                net2[k] = sample(res2.ret - res2.cost)
                d = pd.DataFrame({"n_long": res.n_long, "n_short": res.n_short, "turnover": res.turnover,
                                  "cost": res.cost, "n_cash": res.n_cash, "long": res.long, "short": res.short})
                d = d[(d.index >= pd.Period(C.SAMPLE_START, "M")) & (d.index <= pd.Period(C.SAMPLE_END, "M"))]
                d["factor"] = k
                diag_rows.append(d.reset_index(names="month"))
        F = pd.DataFrame(cols)
        F.index = F.index.astype(str)
        F.to_parquet(OUT / f"{vname}.parquet")
        if vname == "baseline":
            N1 = pd.DataFrame(net); N1.index = N1.index.astype(str); N1.to_parquet(OUT / "baseline_net.parquet")
            N2 = pd.DataFrame(net2); N2.index = N2.index.astype(str); N2.to_parquet(OUT / "baseline_net2x.parquet")
        print(f"{vname:10s} done  T={len(F)}  {time.time() - t0:.0f}s", flush=True)
    if os.environ.get("MOEXZOO_FACTORS_BASELINE_ONLY"):
        diag = pd.concat(diag_rows, ignore_index=True)
        diag["month"] = diag["month"].astype(str)
        diag.to_parquet(OUT / "diagnostics_baseline.parquet")
        print("baseline only", f"{time.time() - t0:.0f}s")
        return
    # point-in-time size/turnover variant: baseline with SIZE->SIZE_PIT, TURN->TURN_PIT
    base = pd.read_parquet(OUT / "baseline.parquet")
    pit = base.copy()
    for k, kp in (("SIZE", "SIZE_PIT"), ("TURN", "TURN_PIT")):
        res = PF.long_short(panel, kp, CH.SIGN[k], min_n=C.MIN_STOCKS_PER_SORT)
        s = sample(res.ret)
        s.index = s.index.astype(str)
        pit[k] = s.reindex(pit.index)
    pit.to_parquet(OUT / "pit.parquet")
    # excluding transition months
    ex = base.copy()
    ex.loc[[m for m in C.TRANSITION_MONTHS if m in ex.index], :] = np.nan
    ex.to_parquet(OUT / "extrans.parquet")
    diag = pd.concat(diag_rows, ignore_index=True)
    diag["month"] = diag["month"].astype(str)
    diag.to_parquet(OUT / "diagnostics_baseline.parquet")
    print("all variants written", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
