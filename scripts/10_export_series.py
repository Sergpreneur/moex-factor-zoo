"""Exports the factor return series used in the paper to CSV (results/factors_csv/): monthly long–short returns
(holding months as rows, the 25 characteristics as columns, decimal returns) for the baseline, every robustness
variant and the net-of-cost series, plus the size factor built in the revision and the baseline series of the
data-construction sensitivities (moexzoo/variants.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402


def main() -> None:
    out = C.RESULTS / "factors_csv"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((C.RESULTS / "factors").glob("*.parquet")):
        if f.stem.startswith("diagnostics"):
            continue
        df = pd.read_parquet(f)
        df.index.name = "holding_month"
        df.to_csv(out / f"{f.stem}.csv", float_format="%.8f")
        n += 1
    for f in sorted(C.RESULTS.glob("revision_factors_*.parquet")):
        df = pd.read_parquet(f)
        df.index.name = "holding_month"
        df.to_csv(out / f"{f.stem}.csv", float_format="%.8f")
        n += 1
    smb = C.RESULTS / "revision_smb_monthly.parquet"
    if smb.exists():
        s = pd.read_parquet(smb).set_index("month")[["smb"]]
        s.index.name = "holding_month"
        s.to_csv(out / "smb_monthly.csv", float_format="%.8f")
        n += 1
    for f in sorted((C.RESULTS / "variants").glob("*/factors/baseline*.parquet")):
        df = pd.read_parquet(f)
        df.index.name = "holding_month"
        df.to_csv(out / f"data_{f.parents[1].name}_{f.stem}.csv", float_format="%.8f")
        n += 1
    (out / "README.md").write_text(
        "Monthly returns (decimal) of the oriented long–short portfolios, one row per holding month.\n\n"
        "* `baseline.csv` — equal-weighted quintile spreads in U100 (Table 3 of the paper).\n"
        "* `baseline_net.csv`, `baseline_net2x.csv` — net of the spread-based cost model (1x, 2x).\n"
        "* `vw`, `u50`, `uall`, `terciles`, `lag1`, `u100ord`, `priceonly`, `extrans`, `pit` — robustness variants.\n"
        "* `revision_factors_*.csv` — exploratory checks of the revision (delisting bounds, suspensions, U100 without\n"
        "  depositary-receipt histories).\n"
        "* `smb_monthly.csv` — the value-weighted size factor built from the sample (revision).\n"
        "* `data_<variant>_baseline*.csv` — baseline series (gross, net, net 2x) under the data-construction\n"
        "  sensitivities of the second revision (`moexzoo/variants.py`): `divcal`, `divcal_dedup`, `yahoo_early`,\n"
        "  `yahoo_late`.\n\n"
        "Columns are the characteristics of Table 2; signs are oriented so that a positive mean confirms the predicted\n"
        "direction. Regenerate with `python3 scripts/10_export_series.py`.\n")
    print(n, "series written to", out)


if __name__ == "__main__":
    main()
