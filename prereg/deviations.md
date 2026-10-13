# Deviations from the pre-analysis plan (v1.0, frozen 2026-10-10T19:41:09Z)

Each entry: what changed, why, whether it was decided before or after seeing results, effect.
This list mirrors Appendix A of the paper.

## A. Implementation details not spelled out in the plan (decided before seeing factor results)

1. **Missing cost estimates.** When the Abdi–Ranaldo estimate is unavailable for a share-month (fewer than
   10 valid two-day pairs), its cost is set to the cross-sectional median cost of that month (50 bp if no
   estimate exists at all). Affects only §7 costs.
2. **ABVOL history requirement.** The code requires ≥ 6 listed months in t−12…t−1 in addition to "both > 0".
   Every universe already requires 12 months of trading history, so the condition never binds.
3. **CAPM-alpha variant (§7 i).** The ladder is run on market-hedged series r_k − β̂_k(MCFTRR − r_f) with
   β̂_k estimated once on the full sample; point estimates and NW t-statistics of α are from the OLS
   regression. The bootstrap therefore ignores sampling error in β̂_k.
4. **Daily returns across long gaps.** A calendar daily return requires a gap of ≤ 14 calendar days between
   sessions (stated in §2 of the plan); the same rule applies to the market index.
5. **Number of clusters.** The plan's §4 table labels nine clusters (Downside separate from Skewness, Size
   separate from Liquidity), while §6 specifies seven for the empirical-Bayes model. The model uses seven;
   a nine-cluster version (exploratory) gives the same replicated set.

## B. Changes made after seeing results

6. **Bug fix — market return after the 2022 suspension.** The MCFTRR series used here has no print on 2022-03-24, so the daily
   market return on 2022-03-25 was computed against the last pre-suspension print (2022-02-25), contrary to
   the 14-day rule. It is now computed against the previous actual print and set to NaN when the gap
   exceeds 14 days. Found by an independent code audit. Effect: characteristics that use daily market
   returns (mainly BETA, ISKEW, COSKEW) change in a few months; baseline t-statistics move by less than
   0.05 and by at most 0.06 in any variant; no discovery count changes in any specification.
7. **Power study design (§7).** Instead of 100 independent outer replications per factor and level, the
   simulation uses 300 common outer resamples (paths of 177 months drawn from the 175 months in which all
   factors are observed) shared across all 25 factors and 6 Sharpe levels. Because the centred bootstrap
   statistics are invariant to the injected drift, the inner Romano–Wolf bootstrap (B = 499) is computed
   once per outer resample. Increases precision; does not change the estimand. (An earlier version padded
   175-month paths with two repeated observations; corrected.)

## C. Exploratory analyses added after seeing results (labelled in the paper)

8. DSR with a bootstrap null benchmark (expected maximum Sharpe ratio under the centred stationary bootstrap).
9. Short-term reversal within turnover terciles of UALL and turnover halves of U100.
10. Empirical Bayes without clusters and with nine clusters.
11. Full ladders within each regime (in addition to the pre-specified difference tests).
12. Two parsimonious Fama–MacBeth specifications (one characteristic per cluster; RW survivors plus STR).
13. Net-of-cost return of a short-term continuation strategy; Romano–Wolf ladder on net returns.
14. Decomposition of the Romano–Wolf critical value (Gaussian statistics with and without the factors'
    correlation matrix; Newey–West t-statistics from i.i.d. normal returns with T = 177, with and without the
    correlation matrix and with L = 4, 8, 12; dispersion and kurtosis of bootstrap statistics; normal scores;
    rescaling to unit variance; five other seeds) and a simulation of family-wise error control (1,000
    replications in each of two designs).
15. Details of the PBO splits (identity of the in-sample winner and its out-of-sample rank).

## D. Exploratory analyses added in the revision (after external comments on the first version, 2026-10-13)

16. Terminal events (delistings) and suspensions: exposure of the portfolio legs; ladders with an assumed delisting
    return of −30%, −50% and −100% charged once to every terminal event, with a one-sided bound (−100% in long legs
    only) and a two-sided bound (−100% in long legs, +50% in short legs), and with suspended shares held until
    trading resumes (`scripts/05e_revision.py`).
17. U100 rebuilt without the six depositary-receipt histories in the data (TCSG, OZON, VKCO, ETLN, MDMG, GEMC).
18. RESMOM and IVOL with a two-factor model (market + value-weighted SMB built from UALL) instead of the CAPM.
19. Implementability of the survivors: long and short legs against all sorted shares, break-even borrow fees, short
    legs replaced by cash during the 2022 short-sale ban (holding months 2022-03..2022-06), short legs restricted to
    U50 or to MOEX Russia index constituents, and an illustrative square-root price-impact charge.
20. Confidence intervals and minimum detectable changes for the regime differences; regime comparison for the
    average of the four momentum survivors.
21. Correlations and spanning regressions among the survivors.
22. The FWER simulation of item 14 enlarged from 1,000 to 4,000 replications per design and extended to Holm and the
    t > 3 rule; a size-calibrated Romano–Wolf level (`scripts/05f_fwer_calibration.py`). The first version reported
    6.2% and 6.5% for the Romano–Wolf rung.
23. Discrepancy with the plan, found in the revision: the plan (§2) states that depositary receipts are not in the
    source and are therefore excluded. In fact the source contains six receipt histories (TCSG, OZON, VKCO, ETLN,
    MDMG, GEMC), which are in the baseline universe (295 U100 stock-months from 2020-09); other receipts are missing
    (see DATA_PROVENANCE.md). The baseline is kept as estimated; the plan-conforming universe without the six
    issuers gives the same five Romano–Wolf survivors (item 17). The first version of the paper repeated the plan's
    statement and described each characteristic as defined as in its source paper; both are corrected (Section 3.2
    and Appendix Table "Implementation relative to the source definitions").

24. Second revision: data-construction sensitivities that rebuild the panels, characteristics and baseline
    portfolios from the raw data in separate folders (`scripts/09_run_data_sensitivities.sh`,
    `scripts/run_variant.py`, `scripts/05g_data_sensitivity.py`; definitions in `moexzoo/variants.py`): (a) a corrected dividend calendar (ex-date on the exchange calendar,
    settlement regime read from the trading board before September 2013, payout credited on the first own trading day
    on or after the ex-date); (b) as (a), and payments of a share that have the same record date treated as possible
    duplicates (only the largest kept). The variants live in a separate module, so `moexzoo/config.py` and
    `moexzoo/data.py` are byte-identical to the versions hashed in `prereg/FREEZE.txt` and the default rules of the plan
    are unchanged. (Of the four hashed files, only `scripts/01_build_panel.py` differs from the freeze, by the bug fix of
    item 6.)
25. Second revision: look-ahead tests for universe membership and trading-cost estimates, a continuity test of
    adjusted share counts through splits (`tests/test_lookahead.py`), and a test that the variant module reproduces the
    frozen adjustment exactly under the plan's dividend rule (`tests/test_variants.py`).
26. Second revision: the index-restricted short leg (item 19) is evaluated only from holding month 2013-01, when the
    first MOEX Russia index base is in force; the first revision used the December 2012 base for earlier months.
27. Second revision: one of the dividend sources (`skill_fill_yahoo`, derived from Yahoo Finance) stores ex-dates,
    whose convention is not uniform before September 2013 (Sberbank: 2012-04-11, the session before the record date of
    2012-04-12; 2013-04-12, the session after the record date of 2013-04-11). The plan rule takes the stored date as the
    ex-date and is unchanged; two further sensitivities of item 24 move every date of this source two exchange sessions
    earlier (`yahoo_early`) or later (`yahoo_late`). The first version of the paper said that all dividend sources store
    the record date; Section 3.3 now describes the sources correctly (the frozen `config.py` already set
    `EX_DATE_SOURCE` to this source, so the code was right and only the text was wrong).
28. Second revision: from 2013-03-25 to 2013-09-01 the price source takes the daily rows of 23 shares (Sberbank, LUKOIL,
    VTB and others) at least partly from the T+2 board, which carried a small part of their trading until the main
    board moved to T+2; their volume-based characteristics are understated in 2013 and abnormal volume is distorted
    until 2014-08. `scripts/05h_board_volume.py` sets the seven volume- and trading-based characteristics of these
    shares to missing for formation months 2013-03 to 2014-08 and reruns the ladder (`results/summary_board_volume.json`).

None of these changes alters a pre-specified result.

Apart from item 23, no characteristic, sign, window, universe, breakpoint, weighting or sample period differs from
the plan.
