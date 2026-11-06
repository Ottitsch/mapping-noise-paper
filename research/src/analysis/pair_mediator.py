"""RQ3 mediator: how many CGR atom rows and CGR edges one swap changes, by location kind.

Rebuilds exactly the swaps the clean models were tested on (same salt, same draws) on the
eligible test reactions of every split with a clean model, and counts the CGR atom rows and
edges whose features differ from the true reaction. Writes results/analysis/pair_mediator.csv
(per split and kind) and pair_mediator_summary.csv (mean over splits).

Usage: python3 -I src/analysis/pair_mediator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "src" / "lib"))
import amn  # noqa: E402
from train import apply_condition  # noqa: E402

G = R / "results" / "grid"
OUT = R / "results" / "analysis"


def counts(a: str, b: str) -> tuple[int, int]:
    Va, Ea = amn.cgr_by_map(a)
    Vb, Eb = amn.cgr_by_map(b)
    rows = sum(Va[m] != Vb.get(m) for m in Va)
    edges = sum(Ea.get(k) != Eb.get(k) for k in set(Ea) | set(Eb))
    return rows, edges


def main():
    recs = []
    for ds in ("rdb7", "e2sn2"):
        df = pd.read_csv(R / "data" / f"{ds}.csv").set_index("id", drop=False)
        for s in range(10):
            tm = G / ds / f"true_s{s}" / "test_metrics.csv"
            if not tm.exists():
                continue
            ids = pd.read_csv(G / ds / f"true_s{s}" / "test_preds.csv.gz").query("cond == 'true'").id.values
            te = df.loc[ids]
            ok = np.array([all(amn.pair_pools(rx).values()) for rx in te.rxn])
            m = pd.read_csv(tm)
            for kind in ("centre", "spect", "mixed"):
                cond = f"pair:{kind}"
                ndraw = int((m.cond == cond).sum())
                for r in range(ndraw):
                    rx, _, _ = apply_condition(te, cond, f"{ds}|{s}|test|{r}", ds)
                    rows, edges = zip(*(counts(a, b) for a, b, e in zip(te.rxn, rx, ok) if e))
                    recs.append(dict(dataset=ds, split=s, kind=kind, draw=r, n=len(rows),
                                     rows=float(np.mean(rows)), edges=float(np.mean(edges)),
                                     unchanged=float(np.mean([x == 0 and y == 0 for x, y in zip(rows, edges)]))))
            print(ds, s, flush=True)
    d = pd.DataFrame(recs)
    d.to_csv(OUT / "pair_mediator.csv", index=False)
    per = d.groupby(["dataset", "kind", "split"])[["rows", "edges", "unchanged"]].mean().reset_index()
    summ = per.groupby(["dataset", "kind"]).agg(splits=("split", "size"), rows=("rows", "mean"), rows_sd=("rows", "std"),
                                                 edges=("edges", "mean"), edges_sd=("edges", "std"),
                                                 unchanged=("unchanged", "mean")).reset_index()
    summ.to_csv(OUT / "pair_mediator_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
