# Raw data provenance (snapshot taken 2026-10-10T19:33Z)

| Folder | Source repository | Commit | Licence | Content used |
|---|---|---|---|---|
| moex_momentum/ | https://github.com/XCemaXX/moex-momentum | 240a69d2f6fcd2009f7a1a65a3bf0fb2b658770e | MIT (Semen Andreev, 2026) | Raw daily MOEX ISS share prices (union of boards), dividends (multi-source; record dates, except the Yahoo-derived source `skill_fill_yahoo`, which stores ex-dates), splits registry, ticker dictionary including delisted securities, MCFTRR index |
| moexutils/ | https://github.com/quantviews/moexutils | 974206079e613c023030b034c82a28e1a65f8770 | see repository | Historical MOEX index calculation bases (number of issued shares, quarterly 2012-12..2026-06), CBR key rate history |
| foykes/ | https://github.com/foykes/moex-dataset | 3320b44df3d825e9c4626377943bdfaf03c019d1 | see repository | MOEX list of securities (11.05.2026) with ISSUE_AMOUNT (current number of shares) |

All market data originate from Moscow Exchange ISS (iss.moex.com) and MOEX index bases; the repositories above are used as public mirrors because iss.moex.com was not reachable from the analysis environment.

## Coverage of depositary receipts

The `moex-momentum` collection contains the full histories, including the receipt periods, of six foreign issuers
whose depositary receipts were later converted into Russian shares under the same ticker (TCSG, OZON, VKCO with the
stitched MAIL history, ETLN, MDMG, GEMC). Receipts that were delisted or replaced by shares with a new ticker are not
in the collection. Eight of them appear in the MOEX Russia index bases (AGRO, FIVE, FIXP, GLTR, HHRU, LNTA, QIWI,
RUALR); at most four index constituents are missing at any quarterly base. Successor shares (X5, HEAD, RAGR, CNRU,
LENT, ...) enter as new securities. `scripts/05e_revision.py` computes these counts and a robustness check without
the six receipt histories.
