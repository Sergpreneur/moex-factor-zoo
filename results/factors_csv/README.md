Monthly returns (decimal) of the oriented long–short portfolios, one row per holding month.

* `baseline.csv` — equal-weighted quintile spreads in U100 (Table 3 of the paper).
* `baseline_net.csv`, `baseline_net2x.csv` — net of the spread-based cost model (1x, 2x).
* `vw`, `u50`, `uall`, `terciles`, `lag1`, `u100ord`, `priceonly`, `extrans`, `pit` — robustness variants.
* `revision_factors_*.csv` — exploratory checks of the revision (delisting bounds, suspensions, U100 without
  depositary-receipt histories).
* `smb_monthly.csv` — the value-weighted size factor built from the sample (revision).
* `data_<variant>_baseline*.csv` — baseline series (gross, net, net 2x) under the data-construction
  sensitivities of the second revision (`moexzoo/variants.py`): `divcal`, `divcal_dedup`, `yahoo_early`,
  `yahoo_late`.

Columns are the characteristics of Table 2; signs are oriented so that a positive mean confirms the predicted
direction. Regenerate with `python3 scripts/10_export_series.py`.
