"""Exploratory (second revision): inference ladder on the data-construction sensitivities built by
scripts/09_run_data_sensitivities.sh (moexzoo/variants.py: corrected dividend calendar; same-date payments treated
as possible duplicates; dates of the ex-date source two sessions earlier or later). Compares survivors and
t-statistics with the pre-specified baseline and counts the changed inputs.

Output: results/summary_data_sensitivity.json
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

_spec = importlib.util.spec_from_file_location("inf04", Path(__file__).resolve().parent / "04_inference.py")
inf04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf04)

VARIANTS = ("divcal", "divcal_dedup", "yahoo_early", "yahoo_late")


def load_f(path: Path) -> pd.DataFrame:
    F = pd.read_parquet(path)
    F.index = pd.PeriodIndex(F.index, freq="M")
    return F[CH.CHAR_IDS]


def main() -> None:
    assert not os.environ.get("MOEXZOO_VARIANT"), "run without MOEXZOO_VARIANT"
    out: dict = {}
    base = load_f(C.RESULTS / "factors" / "baseline.parquet")
    pbase = pd.read_parquet(C.PROCESSED / "panel.parquet")
    lad0, info0 = inf04.ladder(base)
    out["baseline"] = {"survivors": [k for k in CH.CHAR_IDS if lad0.loc[k, "R6_rw"]], "crit": info0["rw_crit_first"],
                       "counts": info0["counts"], "t": {k: float(lad0.loc[k, "t"]) for k in CH.CHAR_IDS}}
    for v in VARIANTS:
        F = load_f(C.ROOT / "results" / "variants" / v / "factors" / "baseline.parquet")
        lad, info = inf04.ladder(F)
        log = json.loads((C.ROOT / "data" / f"interim_{v}" / "build_log.json").read_text())
        pv = pd.read_parquet(C.ROOT / "data" / f"processed_{v}" / "panel.parquet")
        key = ["month", "secid"]
        mm = pbase[key + ["U100", "R_next"]].merge(pv[key + ["U100", "R_next"]], on=key, suffixes=("", "_v"))
        chg = (mm["R_next"] - mm["R_next_v"]).abs() > 1e-12
        same_u = (mm["U100"] == mm["U100_v"]).mean()
        out[v] = {"survivors": [k for k in CH.CHAR_IDS if lad.loc[k, "R6_rw"]], "crit": info["rw_crit_first"],
                  "counts": info["counts"], "t": {k: float(lad.loc[k, "t"]) for k in CH.CHAR_IDS},
                  "max_abs_dt": float(np.nanmax(np.abs(lad["t"].values - lad0["t"].values))),
                  "max_abs_dret_pct": float(100 * np.nanmax(np.abs(F.values - base.values))),
                  "div_moved_all": int(sum(x.get("div_moved", 0) for x in log.values())),
                  "div_dedup_dropped_all": int(sum(x.get("div_dedup_dropped", 0) for x in log.values())),
                  "u100_holding_returns_changed": int((chg & mm["U100"]).sum()),
                  "u100_membership_agreement": float(same_u)}
        print(v, out[v]["survivors"], "crit %.2f" % out[v]["crit"], "max|dt| %.3f" % out[v]["max_abs_dt"],
              "moved", out[v]["div_moved_all"], "dropped", out[v]["div_dedup_dropped_all"],
              "U100 R changed", out[v]["u100_holding_returns_changed"], flush=True)
    # the duplicate treatment alone: divcal_dedup against divcal (divcal_dedup is built on the corrected calendar)
    pa = pd.read_parquet(C.ROOT / "data" / "processed_divcal" / "panel.parquet")
    pb = pd.read_parquet(C.ROOT / "data" / "processed_divcal_dedup" / "panel.parquet")
    key = ["month", "secid"]
    mm = pa[key + ["U100", "R_next"]].merge(pb[key + ["R_next"]], on=key, suffixes=("", "_v"))
    chg = (mm["R_next"] - mm["R_next_v"]).abs() > 1e-12
    ta, tb = pd.Series(out["divcal"]["t"]), pd.Series(out["divcal_dedup"]["t"])
    out["dedup_vs_divcal"] = {"u100_holding_returns_changed": int((chg & mm["U100"]).sum()),
                              "max_abs_dt": float((tb - ta).abs().max())}
    print("dedup vs divcal", out["dedup_vs_divcal"], flush=True)
    (C.RESULTS / "summary_data_sensitivity.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
