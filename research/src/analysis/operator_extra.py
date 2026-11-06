"""Realised nominal and effective fractions for the RDB7 levels 0.6 and 0.8 added after the results
audit (splits 0 to 2, same salts as the training driver), and the dataset size table.
Also the share of reactions where a centre, a spectator and a mixed swap all exist (the RQ3
subset). Writes results/operator/rdb7_fractions_extra.csv, results/operator/datasets.csv.

Usage: python3 -I src/analysis/operator_extra.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import amn  # noqa: E402
import train  # noqa: E402
from train import apply_condition  # noqa: E402

R = HERE.parents[1]
DATA = R / "data"
OUT = R / "results" / "operator"


def main():
    rows = []
    for ds in ("rdb7", "e2sn2", "cyclo"):
        df = pd.read_csv(DATA / f"{ds}.csv")
        elig = sum(all(amn.pair_pools(rx).values()) for rx in df.rxn) / len(df)
        rows.append(dict(dataset=ds, reactions=len(df), heavy_atoms_mean=df.n_heavy.mean(), centre_atoms_mean=df.n_centre.mean(),
                         share_rq3_eligible=elig))
    pd.DataFrame(rows).to_csv(OUT / "datasets.csv", index=False)
    ds = "rdb7"
    df = pd.read_csv(DATA / f"{ds}.csv")
    allsp = json.loads((DATA / f"splits_{ds}.json").read_text())
    rows = []
    for s in range(3):
        sp = allsp[str(s)]
        te = df.iloc[sp["test"]].reset_index(drop=True)
        tr = df.iloc[sp["train"]].reset_index(drop=True)
        for c in ("within:0.6", "within:0.8"):
            for r in range(3):
                _, nom, eff = apply_condition(te, c, f"{ds}|{s}|test|{r}", ds, effective=True)
                rows.append(dict(split=s, part="test", cond=c, draw=r, nominal=nom, effective=eff,
                                 reactions_touched=train.LAST["touched"] / len(te)))
            _, nom, eff = apply_condition(tr, c, f"{ds}|{s}|train", ds, effective=True)
            rows.append(dict(split=s, part="train", cond=c, draw=0, nominal=nom, effective=eff,
                             reactions_touched=train.LAST["touched"] / len(tr)))
        print("split", s, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rdb7_fractions_extra.csv", index=False)


if __name__ == "__main__":
    main()
