"""Data-validation and descriptive numbers quoted in the paper (appended to paper/numbers.tex):
market replication check, survivorship counts, adjustment logs, continuation strategy net of costs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moexzoo import config as C  # noqa: E402
from moexzoo import inference as I  # noqa: E402

NUM: dict[str, str] = {}


def num(k, v, d=2):
    NUM[k] = str(int(v)) if isinstance(v, (int, np.integer)) else f"{v:.{d}f}"


def main():
    m = pd.read_parquet(C.INTERIM / "monthly.parquet")
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    mk = pd.read_parquet(C.INTERIM / "market_monthly.parquet")
    mk.index = pd.PeriodIndex(mk.index, freq="M")
    r = m.pivot(index="month", columns="secid", values="ret").sort_index()
    cap = m.pivot(index="month", columns="secid", values="cap").sort_index()
    src = m.pivot(index="month", columns="secid", values="shares_src").sort_index()
    w = cap.where(src == "index_base_pit").shift(1)
    vw = (w * r).sum(axis=1, min_count=10) / w.where(r.notna()).sum(axis=1, min_count=10)
    cmpd = pd.DataFrame({"vw": vw, "mk": mk["mkt_ret"]}).dropna()
    cmpd = cmpd[(cmpd.index >= pd.Period("2013-01", "M")) & (cmpd.index <= pd.Period(C.SAMPLE_END, "M"))]
    num("ValCorr", float(cmpd.corr().iloc[0, 1]), 3)
    num("ValDiffAnn", float(12 * 100 * (cmpd["vw"] - cmpd["mk"]).mean()), 2)
    num("ValT", len(cmpd))

    tick = json.loads((C.MM / "tickers.json").read_text())
    num("NdictSecids", len(tick))
    num("NdictDelisted", sum(1 for v in tick.values() if v.get("delisted_after")))
    log = json.loads((C.INTERIM / "build_log.json").read_text())
    num("NpriceSecids", len(log))
    num("NbadPrints", sum(len(v.get("bad_prints", [])) for v in log.values()))
    num("NdivUsed", sum(v.get("div_used", 0) for v in log.values()))
    sk = {k: sum(v.get("div_skipped", {}).get(k, 0) for v in log.values()) for k in ("non_rub", "before_first", "after_last", "gap")}
    for k, v in sk.items():
        num("NdivSkip" + k.replace("_", "").capitalize(), int(v))
    num("NsplitSecids", sum(1 for v in log.values() if v.get("splits", 0) > 0))

    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    p = p[(p["month"] + 1 >= pd.Period(C.SAMPLE_START, "M")) & (p["month"] + 1 <= pd.Period(C.SAMPLE_END, "M"))]
    ever_all = set(p.loc[p["UALL"], "secid"])
    ever_100 = set(p.loc[p["U100"], "secid"])
    last_month = pd.Period(C.SAMPLE_END, "M")
    alive_end = set(m.loc[m["month"] == last_month, "secid"])
    # a predecessor in an ISS ticker change whose successor still trades has not left the market
    from moexzoo import universe as UN  # noqa: E402
    succ = {p_: s_ for p_, s_, _ in UN.rename_edges(tick)}

    def alive(s_, depth=0):
        if s_ in alive_end:
            return True
        return depth < 10 and s_ in succ and alive(succ[s_], depth + 1)
    dead_all = {s_ for s_ in ever_all if not alive(s_)}
    dead_100 = {s_ for s_ in ever_100 if not alive(s_)}
    num("NeverAll", len(ever_all))
    num("NeverHundred", len(ever_100))
    num("NdeadAll", len(dead_all))
    num("NdeadHundred", len(dead_100))
    num("NdeadHundredPct", 100 * len(dead_100) / len(ever_100), 0)
    num("NrenamedHundred", len({s_ for s_ in ever_100 if s_ not in alive_end and alive(s_)}))
    num("PanelRowsHundred", int(p["U100"].sum()))
    num("PanelRowsAll", int(p["UALL"].sum()))
    num("NformMonths", p["month"].nunique())

    # continuation (= -STR) net of costs, back-of-envelope (exploratory)
    d = pd.read_parquet(C.RESULTS / "factors" / "diagnostics_baseline.parquet")
    d = d[d["factor"] == "STR"].set_index("month")
    F = pd.read_parquet(C.RESULTS / "factors" / "baseline.parquet")
    cont = (-F["STR"] - d["cost"].reindex(F.index)).dropna()
    ts = I.tstats(pd.DataFrame({"cont_net": cont}))
    num("ContNetMean", 100 * float(ts["mean"].iloc[0]))
    num("ContNetT", float(ts["t"].iloc[0]))
    num("ContTurnover", float(d["turnover"].mean()))
    # momentum crash months
    for k in ("MOM12_1", "HIGH52", "MOM6_1"):
        s = F[k]
        num("Worst" + k.replace("_", ""), 100 * float(s.min()), 1)
        num("WorstAbs" + k.replace("_", ""), abs(100 * float(s.min())), 1)
        NUM["WorstMonth" + k.replace("_", "")] = str(s.idxmin())
    # share of payers and characteristic coverage in U100
    out = ["% Appended by scripts/07_validation_stats.py"]
    for k, v in sorted(NUM.items()):
        out.append("\\expandafter\\def\\csname val@" + k + "\\endcsname{" + v + "}")
    path = C.ROOT / "paper" / "numbers_validation.tex"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(f"{k} = {v}" for k, v in sorted(NUM.items())))


if __name__ == "__main__":
    main()
