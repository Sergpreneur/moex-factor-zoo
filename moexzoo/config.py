"""Single source of truth for every tunable constant of the study.

Values marked [PAP] are fixed by the pre-analysis plan (prereg/analysis_plan.md).
Changing any of them after the plan hash was recorded is a deviation and must be
logged in prereg/deviations.md.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MM = RAW / "moex_momentum"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"

# --- Market microstructure conventions (MOEX) -------------------------------------
# Settlement regime decides how far before the record date a share goes ex-dividend.
SETTLEMENT_T2_FROM = "2013-09-02"  # T0 -> T+2 on the main board
SETTLEMENT_T1_FROM = "2023-07-31"  # T+2 -> T+1
EX_DATE_SOURCE = "skill_fill_yahoo"  # this dividend source stores the ex-date itself
MIN_STOCKS_FOR_CALENDAR_DAY = 20  # a weekday/weekend date with >= 20 trading shares is a session
MAX_SESSION_GAP_DAYS = 14  # longer gaps (2022 closure) do not produce a daily return

# --- Sample [PAP] -----------------------------------------------------------------
DAILY_START = "2006-01-01"  # daily history used to build characteristics
FIRST_FORMATION = "2011-12"  # first portfolio-formation month (end of month)
SAMPLE_START = "2012-01"  # first holding month
SAMPLE_END = "2026-09"  # last holding month
REGIME_BREAK = "2022-03"  # first holding month of the post-2022 ("closed market") regime
TRANSITION_MONTHS = ("2022-02", "2022-03")  # dropped in one robustness variant

# --- Universe [PAP] ---------------------------------------------------------------
MIN_TRADING_DAYS_IN_MONTH = 10  # trading days with positive volume in month t, capped at half the month's sessions
LAST_TRADE_WITHIN_DAYS = 5  # last trade within the final 5 sessions of month t
HISTORY_MONTHS_FOR_LIQ = 12  # trailing months used to rank liquidity
TOP_N_BASELINE = 100  # U100: 100 most liquid shares by median monthly value
TOP_N_LARGE = 50  # U50 robustness universe
MIN_STOCKS_PER_SORT = 25  # a factor-month needs >= 25 names with the characteristic
N_GROUPS = 5  # quintiles [PAP]; terciles in robustness

# --- Daily data screens [PAP] -----------------------------------------------------
BAD_PRINT_UP = 1.0  # daily total return above +100% ...
BAD_PRINT_DOWN = -0.6  # ... or below -60% ...
BAD_PRINT_REVERSAL_BAND = 0.2  # ... that reverses within +-20% over two days is set to NaN

# --- Characteristic windows [PAP] -------------------------------------------------
DAYS_12M = 252
MIN_OBS_12M = 120
MIN_OBS_1M = 15
CAP_WEIGHT_PERCENTILE = 0.80  # capped value weights (Jensen, Kelly & Pedersen 2023)

# --- Inference [PAP] --------------------------------------------------------------
ALPHA = 0.05
HLZ_HURDLE = 3.0
BOOT_REPS = 10_000
BOOT_MEAN_BLOCK = 6  # stationary bootstrap mean block length (months)
BOOT_SEED = 20261010
DSR_PASS = 0.95
CSCV_SPLITS = 16
