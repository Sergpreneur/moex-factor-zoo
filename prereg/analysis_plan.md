# Pre-Analysis Plan — "How Many Factors Are Real on the Moscow Exchange?"

A multiple-testing replication of the Russian equity factor zoo

Author: Sergey Nefedov (Lomonosov Moscow State University)
Plan version: 1.0 — frozen 2026-10-10, before any characteristic-sorted portfolio return was computed.
Freeze record: `prereg/FREEZE.txt` (SHA-256 of this file and of `moexzoo/config.py`, UTC timestamp).

## 0. What had been looked at before the freeze

Only data-quality diagnostics that do not condition on any characteristic:
(i) number of listed shares, trading sessions and dividend records per year;
(ii) the unconditional distribution of next-month returns of the 100 most liquid shares and a list of
their |monthly return| > 60–100 % cases (all traced to genuine events, e.g. MSST Aug-2018);
(iii) a check that a value-weighted portfolio built from MOEX index-base share counts tracks the
MCFTRR total-return index (monthly correlation 0.988, 2013–2026);
(iv) coverage of share-count data in the liquid universe.
No characteristic was computed and no portfolio was sorted before this plan was frozen.
Factor definitions are taken from the cited source papers, not chosen on this data.
This is an internal, time-stamped pre-specification; it was not filed with an external registry.

## 1. Questions and hypotheses

Q1. Which of the 25 price-, trading- and payout-based return predictors documented in the international
literature produce a premium on MOEX, and how does the count depend on the multiple-testing correction?

Q2. Did the premia change when non-resident investors were locked out of the market in February 2022?

Hypotheses (k indexes characteristics; μ_k is the mean monthly return of the long–short portfolio oriented
so that the source paper predicts μ_k > 0):

- H1 (individual premia): μ_k > 0. Evaluated separately under every rung of the ladder in §6.
- H2 (global null): μ_k = 0 for all k, tested against "some μ_k ≠ 0" by the max-|t| bootstrap
  (single-step Romano–Wolf critical value).
- H3 (multiplicity): the number of discoveries is non-increasing along the ladder; we report by how much it
  falls between the conventional rung and the dependence-robust FWER rung. Descriptive.
- H4 (regime): μ_k,post ≠ μ_k,pre for each k (two-sided), and the equal-weighted average of the 25 oriented
  premia differs across regimes. Exploratory: the post-regime sample is short and we expect low power.
- H5 (tradability): for every characteristic that survives the Romano–Wolf rung, the mean return net of
  estimated trading costs is positive.

## 2. Data

Source: Moscow Exchange ISS data mirrored in public repositories (see `data/raw/PROVENANCE.md`):
raw daily OHLC, volume and RUB value for all shares ever traded on the MOEX shares market
(survivorship-free ticker dictionary, 1,031 SECIDs incl. 758 delisted), dividends (record dates, multiple
sources reconciled upstream), splits, MCFTRR (MOEX total-return index, net of 13 % dividend tax), MOEX
index calculation bases 2012-12…2026-06 (number of issued shares), MOEX list of securities 2026-05-11
(current number of shares), CBR key rate (refinancing rate before 2013-09).

Depositary receipts are not in the source and are therefore excluded. Ordinary and preferred shares
(including shares of foreign issuers traded on the shares market) are included.

Adjustments (code: `moexzoo/data.py`):
- Splits: prices before a split date multiplied by before/after; volumes divided; dividend amounts with
  record date before the split multiplied likewise.
- Dividends: gross (no tax). Ex-date = own trading day located by settlement regime relative to the record
  date: T0 before 2013-09-02 (ex-date = next session after the record date), T+2 until 2023-07-30 (one
  session before), T+1 from 2023-07-31 (record date itself); one source stores the ex-date directly.
  Non-RUB payments are dropped. A payout whose implied ex-date leaves the record month ±1 is dropped.
- Daily total return on own trading days r = (P + D)/P_prev − 1; a calendar daily return exists only if the
  share traded on the previous session and the gap is ≤ 14 calendar days (drops the 2022-02-25 → 2022-03-24
  closure from daily statistics; monthly returns still contain it).
- Screen: a single print with daily return > +100 % or < −60 % that reverts to within ±20 % of the prior
  price over two days is deleted.
- Monthly return = ratio of the total-return index at the last trade of month t+1 to that at the last trade
  of month t; NaN if the share did not trade in either month.

Sample: formation months 2011-12 … 2026-08; holding months 2012-01 … 2026-09 (T = 177).
Daily data from 2006 are used only to build trailing characteristics. Dividend coverage before 2010 is thin,
so characteristics with windows reaching back before 2011 (LTR, SEAS, RESMOM in early months) are partly
price-based; this is reported, not corrected.

Market return: MCFTRR. Risk-free: key rate in force on the first day of the month, compounded monthly.

## 3. Universes (formation at the last session of month t)

Activity filter (all universes): the share traded on at least min(10, ⌈½ · sessions in month t⌉) sessions
in month t, its last trade lies within the final 5 sessions of month t, it has 12 months of trading-value
history, and it is not marked delisted before the end of month t. For ISS ticker changeovers (renames),
the predecessor is used before the rename month and the successor from the rename month on (no double
counting). Redomiciled securities are different instruments and are not stitched.

- U100 (baseline): the 100 most liquid active shares by the median of monthly RUB trading value over months
  t−11…t (relative cut; stable breadth over time).
- U50 (robustness): the 50 most liquid.
- UALL (robustness): all active shares.
- U100-ORD (robustness): U100 after dropping a preferred share whenever an ordinary share of the same
  issuer is in U100 that month (preferred-only issuers such as Transneft stay).

## 4. Characteristics (25; predicted sign in brackets; 1M = sessions of month t; 12M = last 252 sessions)

| ID | Cluster | Definition | Min. data | Sign | Source |
|---|---|---|---|---|---|
| MOM12_1 | Momentum | ∏(1+R) over months t−11…t−1 − 1 | all 11 | + | Jegadeesh & Titman (1993) |
| MOM6_1 | Momentum | months t−5…t−1 | all 5 | + | Jegadeesh & Titman (1993) |
| MOM12_7 | Momentum | months t−11…t−6 | all 6 | + | Novy-Marx (2012) |
| RESMOM | Momentum | Σ CAPM residuals t−11…t−1 / sd of those residuals; monthly excess returns regressed on MCFTRR excess over t−35…t−1 | ≥24 incl. all of t−11…t−1 | + | Blitz, Huij & Martens (2011) |
| HIGH52 | Momentum | close_t / max close over 12M | ≥120 trades | + | George & Hwang (2004) |
| STR | Reversal | R_t | — | − | Jegadeesh (1990) |
| LTR | Reversal | ∏(1+R) over months t−35…t−12 − 1 | ≥20 of 24 | − | De Bondt & Thaler (1985) |
| SEAS | Seasonality | mean of R in months t+1−12j, j = 1…5 | ≥2 | + | Heston & Sadka (2008) |
| BETA | Low risk | Dimson beta b0+b1 from r_d = a + b0 m_d + b1 m_{d−1}, 12M | ≥120 | − | Frazzini & Pedersen (2014) |
| IVOL | Low risk | sd of residuals of r_d on m_d, 1M | ≥15 | − | Ang, Hodrick, Xing & Zhang (2006) |
| TVOL | Low risk | sd of r_d, 1M | ≥15 | − | Ang, Hodrick, Xing & Zhang (2006) |
| MAX | Low risk | max r_d, 1M | ≥15 | − | Bali, Cakici & Whitelaw (2011) |
| DBETA | Downside | cov(r,m | m<m̄)/var(m | m<m̄), 12M | ≥120, ≥40 down days | + | Ang, Chen & Xing (2006) |
| SKEW | Skewness | skewness of r_d, 12M | ≥120 | − | Boyer, Mitton & Vorkink (2010); Amaya et al. (2015) |
| ISKEW | Skewness | skewness of market-model residuals, 12M | ≥120 | − | Boyer, Mitton & Vorkink (2010) |
| COSKEW | Skewness | E[ε_i ε_m²] / (√E[ε_i²] · E[ε_m²]), ε_i market-model residual, ε_m demeaned m, 12M | ≥120 | − | Harvey & Siddique (2000) |
| SIZE | Size | log market cap at t | cap available | − | Banz (1981) |
| AMIHUD | Liquidity | mean |r_d| / value_d (RUB mn), 12M | ≥120 | + | Amihud (2002) |
| DVOL | Liquidity | log mean daily value over 12M (no-trade sessions = 0) | ≥120 trades | − | Brennan, Chordia & Subrahmanyam (1998) |
| TURN | Liquidity | Σ value over months t−11…t / mean month-end cap over t−11…t | ≥6 caps | − | Datar, Naik & Radcliffe (1998) |
| ZERO | Liquidity | share of 12M sessions (since first trade, ≥120) with no trade or zero return | ≥120 sessions | + | Lesmond, Ogden & Trzcinka (1999); Bekaert, Harvey & Lundblad (2007) |
| SPREAD | Liquidity | mean Corwin–Schultz two-day spread (negatives → 0, overnight-adjusted), 1M | ≥12 pairs | + | Corwin & Schultz (2012) |
| CVVAL | Liquidity | sd/mean of monthly value over t−11…t | 12 months | − | Chordia, Subrahmanyam & Anshuman (2001) |
| ABVOL | Liquidity | log(mean daily value in t / mean daily value over t−12…t−1) | both > 0 | + | Gervais, Kaniel & Mingelgrin (2001) |
| DY | Payout | Σ gross dividends with ex-date in the 12 months to t / close_t; sorted among payers only | DY > 0 | + | Litzenberger & Ramaswamy (1979) |

Market caps: point-in-time share counts from MOEX index bases when the share is a constituent; otherwise
the nearest index-base count of the same SECID; otherwise the current (2026-05-11) number of shares;
otherwise missing. Share counts are put on the split-adjusted scale. Caps therefore contain look-ahead for
non-constituents; SIZE, TURN and value weights are interpreted with this caveat, and a robustness variant
restricts SIZE/TURN to point-in-time share counts.

## 5. Portfolios

At the last session of month t, within the universe, shares with a non-missing characteristic are sorted
into quintiles (ranks, ties broken by SECID); a factor-month needs ≥ 25 shares. Equal weights (baseline).
Long–short = top minus bottom quintile, oriented by the predicted sign. Held over month t+1, monthly
rebalancing. A held share that does not trade in month t+1 earns 0 (cash) that month. Returns are gross,
before costs and taxes.

## 6. Inference — the ladder

Statistic: mean monthly long–short return; t-statistic with Newey–West standard errors, lag
L = ⌊4(T/100)^{2/9}⌋ (= 4 for T = 177). Two-sided p-values. A *discovery* is a rejection with t > 0
(predicted direction); wrong-signed rejections are reported separately.

- R1 conventional: t > 1.96.
- R2 Harvey–Liu–Zhu hurdle: t > 3.0.
- R3 Holm (FWER 5 %).
- R4 Benjamini–Hochberg (FDR 5 %).
- R5 Benjamini–Yekutieli (FDR 5 %, arbitrary dependence).
- R6 Romano–Wolf stepdown (FWER 5 %), studentised, two-sided, joint stationary bootstrap of months
  (mean block 6, B = 10,000, seed 20261010), NW standard errors recomputed in each resample, NaN-aware for
  factors with missing months; adjusted p-values by Romano & Wolf (2016). Centrepiece rung.
- R7 Deflated Sharpe Ratio > 0.95 (Bailey & López de Prado 2014), N = 25 trials, V[SR] = cross-sectional
  variance of the 25 monthly Sharpe ratios, skewness/kurtosis of each series. Sensitivity: N = N_eff and
  N = number of configurations in the full robustness grid (§7).
- Family-level: Probability of Backtest Overfitting by CSCV (16 blocks, Sharpe metric) for selecting the
  in-sample best of the 25 factors (Bailey, Borwein, López de Prado & Zhu 2017).
- Effective number of tests N_eff from the eigenvalues of the factor-return correlation matrix
  (Nyholt 2004; Li & Ji 2005).
- Empirical Bayes (complement, after Jensen, Kelly & Pedersen 2023): μ̂ = μ + ε, ε ~ N(0, Σ/T);
  μ_k = μ0 + c_cluster(k) + w_k with cluster effects (7 clusters of §4) and idiosyncratic effects; μ0, τ_c,
  τ_w by maximum marginal likelihood; report posterior means, 95 % intervals, P(μ_k > 0), and the share of
  factors whose interval excludes zero on the positive side.

## 7. Robustness grid (all reported)

(a) capped value weights (cap at the 80th percentile of caps in the universe-month; shares without cap
excluded); (b) U50; (c) UALL; (d) terciles; (e) one-session implementation lag (enter at the close of the
first trade of month t+1, at most the 3rd session; otherwise cash); (f) U100-ORD; (g) price-only holding
returns; (h) excluding holding months 2022-02 and 2022-03; (i) CAPM alpha instead of mean (MCFTRR excess);
(j) SIZE and TURN with point-in-time share counts only. Each variant is run through the full ladder.
Bootstrap block length 3 and 12 for R6. Configurations counted for the DSR sensitivity: 25 × 10 = 250.

Regimes (H4): pre = holding months 2012-01…2022-02, post = 2022-03…2026-09. For each k, Δ_k from
r_t = a + Δ·1{post} + e with NW (L = 4); BH across k; the same for the equal-weighted average of the 25
oriented factors.

Costs (H5): one-way cost per share-month = ½ × Abdi–Ranaldo (2017) close-high-low spread estimate for
month t (negatives → 0) + 5 bp commission/fees; turnover from drifted weights; net mean, net t and break-even
one-way cost reported for every factor; a 2× cost scenario.

Fama–MacBeth (descriptive): monthly cross-sectional OLS of R_{t+1} on cross-sectional ranks of each
characteristic scaled to [−0.5, 0.5] and oriented; univariate for every k; multivariate with all 25 (missing
ranks set to 0); NW t-statistics.

Power: stationary-bootstrap simulation from the demeaned 25-factor return matrix, drift injected into one
factor at a time (all 25 in turn) with annualised Sharpe ratio ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5};
rejection rates of R1–R6 (R6 with B = 499 inner resamples, 100 outer replications per factor and level).

## 8. Commitments

We will not change any definition, window, sign, breakpoint, universe or sample period in §2–§7 after
seeing results; we will report all 25 characteristics under all rungs and all variants; we will not add
characteristics to the headline family; any departure from this plan is listed with its reason and effect in
`prereg/deviations.md` and in the paper's appendix.

## 9. Anticipated limitations

No accounting data (value, profitability, investment, accruals families are absent). Share counts are
point-in-time only for index constituents. Dividend coverage is thin before 2010. Depositary receipts are
absent. MCFTRR is a net-of-tax index while stock returns are gross. The post-2022 sample is short.
