"""The data-construction sensitivities (moexzoo/variants.py) must leave everything except the dividend rule as in the
frozen adjustment of moexzoo/data.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402
from moexzoo import data as D  # noqa: E402
from moexzoo import variants as V  # noqa: E402

SECIDS = ("SBER", "SBERP", "LKOH", "NLMK", "CHMF", "VTBR", "GMKN", "HYDR", "MTSS", "SELGP")


def test_variant_adjustment_reproduces_frozen_one():
    """With the plan's dividend rule, the variant copy of adjust_security equals data.adjust_security for every
    security in the data."""
    V.VARIANTS["_identity"] = dict(rule="plan", dedup=False, ex_shift=0)
    old = V.ACTIVE
    V.ACTIVE = "_identity"
    try:
        for s in D.list_secids():
            a, b = D.adjust_security(s), V.adjust_security(s)
            if a is None:
                assert b is None
                continue
            pd.testing.assert_frame_equal(a.daily, b.daily, check_exact=True)
            assert b.log["div_moved"] == 0
    finally:
        V.ACTIVE = old
        V.VARIANTS.pop("_identity")


def test_yahoo_shift_moves_only_that_source():
    old = V.ACTIVE
    V.ACTIVE = "yahoo_late"
    try:
        for s in SECIDS:
            divs = D.load_dividends(s)
            a, b = D.adjust_security(s), V.adjust_security(s)
            if not (divs["source"].astype(str) == C.EX_DATE_SOURCE).any():
                pd.testing.assert_frame_equal(a.daily, b.daily, check_exact=True)
            # prices and volumes are never touched
            pd.testing.assert_frame_equal(a.daily[["close", "volume", "value"]], b.daily[["close", "volume", "value"]])
    finally:
        V.ACTIVE = old


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        V.activate("no_such_variant")
    assert C.INTERIM.name == "interim" and D.adjust_security.__module__ == "moexzoo.data"
