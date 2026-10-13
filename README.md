# How Many Factors Are Real on the Moscow Exchange?

Replication package for: Sergey Nefedov (2026), *How Many Factors Are Real on the Moscow Exchange?
A Pre-Specified Multiple-Testing Study of the Russian Factor Zoo*. Working paper, Lomonosov Moscow
State University.

25 price-, trading- and payout-based return predictors on MOEX shares (holding months 2012:01–2026:09; share
data free of survivorship bias, depositary receipts covered only in part), evaluated under a ladder of seven
significance criteria
(t > 1.96, Harvey–Liu–Zhu t > 3, Holm, Benjamini–Hochberg, Benjamini–Yekutieli, Romano–Wolf stepdown with a
stationary bootstrap, Deflated Sharpe Ratio), plus PBO, effective number of tests, empirical Bayes, a
pre-specified robustness grid, regime comparison around the 2022 lockout, trading costs and a power study.

## Pre-specification

* `prereg/analysis_plan.md` — the pre-analysis plan, frozen before any characteristic was computed.
* `prereg/FREEZE.txt` — UTC timestamp and SHA-256 hashes of the plan and of the data-construction code at the
  freeze; the freeze is commit `6ec1767` (2026-10-10T19:41:15Z) of the enclosed git history.
* `prereg/deviations.md` — every departure from the plan, with timing and effect (mirrors Appendix A), including the
  exploratory analyses added in the revisions (section D).

Of the four files hashed in `FREEZE.txt`, the plan, `moexzoo/config.py` and `moexzoo/data.py` are unchanged since the
freeze; `scripts/01_build_panel.py` differs by the bug fix logged as deviation 6 (Appendix A, item vi). Check with
`sha256sum prereg/analysis_plan.md moexzoo/config.py moexzoo/data.py scripts/01_build_panel.py`.

This is an internal, time-stamped pre-specification, not a filing with an external registry.

## Layout

| Path | Content |
|---|---|
| `moexzoo/` | library: `config.py` (all constants), `data.py` (adjustments), `universe.py`, `chars.py` (25 characteristics), `portfolios.py`, `inference.py` (NW, Holm, BH/BY, stationary bootstrap, Romano–Wolf, DSR, PBO, N_eff, empirical Bayes); `revision.py` (helpers for the first revision) and `variants.py` (data-construction sensitivities of the second revision) |
| `scripts/00_fetch_data.sh` | downloads the raw data from public repositories at pinned commits |
| `scripts/01_build_panel.py` … `10_export_series.py` | pipeline, run in order (see below); 05e–05h and 09 are the exploratory analyses added in the revisions; `run_variant.py` runs a pipeline script under a data-construction variant |
| `tests/` | integrity tests (look-ahead perturbation of characteristics, universes and cost estimates; splits and share counts; dividends; alignment; inference vs statsmodels; FWER; revision helpers; data-construction variants vs the frozen adjustment) |
| `results/` | all outputs: factor returns (`results/factors/*.parquet`, CSV copies in `results/factors_csv/`), ladders, summaries; `results/variants/` holds the data-construction sensitivities |
| `paper/` | LaTeX source (`main.tex`), bibliography, generated tables, figures and number macros, compiled `main.pdf`; `paper/response/` holds the responses to the two rounds of comments (Russian; `numbers_ru_round1.tex` freezes the numbers of the first response as sent) |
| `DATA_PROVENANCE.md` | data sources, commits and licences |

## Reproducing

Requirements: Python ≥ 3.11 with pandas, numpy, scipy, statsmodels, pyarrow, openpyxl, matplotlib,
adjustText, pytest; TeX Live with XeLaTeX, polyglossia, fontspec, TeX Gyre Termes and FreeSerif.

```bash
bash scripts/00_fetch_data.sh                 # raw data -> data/raw
python3 scripts/01_build_panel.py             # adjusted daily and monthly panels (~0.5 min)
python3 scripts/02_characteristics.py         # universes, 25 characteristics, holding returns (~1 min)
python3 scripts/03_factors.py                 # long–short returns, all variants (~5 min)
python3 scripts/04_inference.py               # ladder, N_eff, PBO, EB, regimes, costs (~0.5 min)
python3 scripts/05_extensions.py              # Fama–MacBeth, power, exploratory checks (~1 min)
python3 scripts/05b_exploratory_fm.py         # exploratory Fama–MacBeth, STR by turnover
python3 scripts/05c_mechanism.py              # critical-value diagnostics, FWER simulation, PBO details (~3 min)
python3 scripts/05d_mechanism_decomp.py       # critical-value decomposition (~2 min)
python3 scripts/05e_revision.py               # revision: delisting bounds, receipts, two-factor, implementability (~6 min)
python3 scripts/05f_fwer_calibration.py       # revision: FWER simulation (4,000 x 2) and calibrated Romano–Wolf (~9 min)
bash scripts/09_run_data_sensitivities.sh     # revision: four dividend-data sensitivities in separate folders (~12 min)
python3 scripts/05g_data_sensitivity.py       # revision: ladders for those sensitivities
python3 scripts/05h_board_volume.py           # revision: volume characteristics around the 2013 board switch (~1 min)
python3 scripts/06_tables_figures.py          # all tables, figures and paper/numbers.tex
python3 scripts/07_validation_stats.py        # paper/numbers_validation.tex
python3 scripts/08_response_numbers.py        # paper/response/numbers_ru.tex (numbers for the response letters)
python3 scripts/10_export_series.py           # factor return series as CSV (results/factors_csv/)
python3 -m pytest -q tests
(cd paper && xelatex main && bibtex main && xelatex main && xelatex main)
(cd paper/response && for f in response response_round2; do xelatex $f && xelatex $f; done)   # responses (Russian)
```

The data-construction sensitivities (`moexzoo/variants.py`: `divcal`, `divcal_dedup`, `yahoo_early`, `yahoo_late`)
are selected with the environment variable `MOEXZOO_VARIANT` and run through `scripts/run_variant.py`, which redirects
all inputs and outputs to `data/interim_<v>`, `data/processed_<v>` and `results/variants/<v>`; the frozen modules
are not modified, and an unknown variant name is an error. For example:
`MOEXZOO_VARIANT=divcal python3 scripts/run_variant.py scripts/01_build_panel.py`. `MOEXZOO_FACTORS_BASELINE_ONLY=1`
makes `03_factors.py` compute only the baseline series.

Most numbers in the text of the paper are macros generated by scripts 06–07. The remaining hand-typed values
(a few t-statistics read off Table 10, ranges that summarise tables, institutional facts and figures quoted
from the literature) were checked against the results files and sources.

## Data

All market data originate from the Moscow Exchange ISS and MOEX index calculation bases, taken from public
mirrors because the ISS endpoints were not reachable from the analysis environment. Share data are free of
survivorship bias; depositary receipts are covered only in part. See `DATA_PROVENANCE.md`.

## Citation and licence

Code and derived results: MIT licence (`LICENSE`). Please cite the paper; `CITATION.cff` holds the reference and
`.zenodo.json` the metadata for archiving a release with a DOI. Raw data are not redistributed; their sources and
licences are listed in `DATA_PROVENANCE.md`.
