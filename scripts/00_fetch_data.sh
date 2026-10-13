#!/usr/bin/env bash
# Fetch the raw data used in the paper from public repositories at the pinned commits
# and place the needed files under data/raw/. Requires git and network access to github.com.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
TMP="$(mktemp -d)"
mkdir -p "$RAW/moex_momentum" "$RAW/moexutils" "$RAW/foykes" "$ROOT/data/interim" "$ROOT/data/processed"

# 1) Survivorship-free MOEX ISS prices, dividends, splits, ticker dictionary, MCFTRR (MIT licence)
git clone -q https://github.com/XCemaXX/moex-momentum.git "$TMP/mm"
git -C "$TMP/mm" checkout -q 240a69d2f6fcd2009f7a1a65a3bf0fb2b658770e
cp -r "$TMP/mm/data/prices_iss" "$TMP/mm/data/dividends" "$TMP/mm/data/splits" "$TMP/mm/data/indices" \
      "$TMP/mm/data/tickers.json" "$TMP/mm/data/tickers_manual.json" "$TMP/mm/data/tickers_unavailable.jsonl" \
      "$RAW/moex_momentum/"
cp "$TMP/mm/LICENSE" "$RAW/moex_momentum/LICENSE"

# 2) MOEX index calculation bases (number of issued shares) and CBR key rate
git clone -q --filter=blob:none --no-checkout https://github.com/quantviews/moexutils.git "$TMP/mu"
for f in stock-index-base.xlsx key_rate.csv sectors.csv; do
  git -C "$TMP/mu" show 974206079e613c023030b034c82a28e1a65f8770:metadata/$f > "$RAW/moexutils/$f"
done

# 3) MOEX list of securities (current number of shares)
git clone -q --filter=blob:none --no-checkout https://github.com/foykes/moex-dataset.git "$TMP/fk"
git -C "$TMP/fk" show 3320b44df3d825e9c4626377943bdfaf03c019d1:datasets/ticker_lists/moex_stocks.csv > "$RAW/foykes/moex_stocks.csv"

cp "$ROOT/DATA_PROVENANCE.md" "$RAW/PROVENANCE.md" 2>/dev/null || true
rm -rf "$TMP"
echo "Raw data written to $RAW"
