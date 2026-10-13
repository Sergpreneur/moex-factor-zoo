#!/usr/bin/env bash
# Exploratory data-construction sensitivities added in the second revision (prereg/deviations.md, items 24 and 27;
# definitions in moexzoo/variants.py). Each variant rebuilds the panels from the raw data in its own folders
# (data/interim_<v>, data/processed_<v>, results/variants/<v>) and computes the baseline factor returns;
# scripts/05g_data_sensitivity.py then runs the inference ladder on them. The pre-specified outputs and the
# frozen modules moexzoo/config.py and moexzoo/data.py are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."
for v in divcal divcal_dedup yahoo_early yahoo_late; do
  echo "== variant $v"
  MOEXZOO_VARIANT=$v python3 scripts/run_variant.py scripts/01_build_panel.py > /dev/null
  MOEXZOO_VARIANT=$v python3 scripts/run_variant.py scripts/02_characteristics.py > /dev/null
  MOEXZOO_VARIANT=$v MOEXZOO_FACTORS_BASELINE_ONLY=1 python3 scripts/run_variant.py scripts/03_factors.py
done
