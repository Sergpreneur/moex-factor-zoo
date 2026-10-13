"""Run a pipeline script under an exploratory data-construction variant (moexzoo/variants.py).

    MOEXZOO_VARIANT=divcal python3 scripts/run_variant.py scripts/01_build_panel.py

The variant is activated before the script runs: the interim, processed and results folders are redirected to
data/interim_<v>, data/processed_<v> and results/variants/<v>, and the dividend rule is replaced. The frozen
modules moexzoo/config.py and moexzoo/data.py are not modified. Used by scripts/09_run_data_sensitivities.sh.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import variants as V  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    name = os.environ.get("MOEXZOO_VARIANT", "")
    if not name:
        raise SystemExit("set MOEXZOO_VARIANT to one of: " + ", ".join(V.VARIANTS))
    V.activate(name)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
