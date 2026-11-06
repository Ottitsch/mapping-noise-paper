"""Quantities added in the first revision that do not need the per split machinery of analyse.py.

1. Wrong heavy atoms per touched test reaction for the uniform levels and for RXNMapper and matched
   maps, per dataset, mean over splits and draws (from the realised fractions and the test sets).
2. The swap mix of the uniform operator at level 0.10 on every RDB7 reaction (one swap or two).
3. The effective fraction at the top level and for the cross element shuffle (coverage of the
   endpoint comparison).

Usage: python3 -I operator_counts.py   (writes results/analysis/operator_counts.json)
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2]
OP = R / "results" / "operator"
out = {}
for ds in ("rdb7", "e2sn2", "cyclo"):
    f = OP / f"{ds}_fractions.csv"
    if not f.exists():
        continue
    t = pd.read_csv(f)
    x = OP / f"{ds}_fractions_extra.csv"
    if x.exists() and x.stat().st_size > 1:
        t = pd.concat([t, pd.read_csv(x)], ignore_index=True)
    t = t[(t.part == "test") & (t.draw < 3)]
    data = pd.read_csv(R / "data" / f"{ds}.csv").set_index("id")
    sp = json.loads((R / "data" / f"splits_{ds}.json").read_text())
    heavy = {int(s): data.iloc[sp[str(s)]["test"]].n_heavy.mean() for s in t.split.unique()}
    t = t.assign(wrong_per_touched=[r.nominal * heavy[r.split] / r.reactions_touched if r.reactions_touched > 0 else float("nan")
                                    for r in t.itertuples()])
    g = t.groupby("cond")[["nominal", "effective", "reactions_touched", "wrong_per_touched"]].mean()
    g["n_splits"] = t.groupby("cond").split.nunique()
    out[ds] = {c: {k: float(v) for k, v in row.items()} for c, row in g.iterrows()}
w = pd.read_csv(R / "results" / "figures" / "rdb7_wrong_counts.csv")
vc = w.uniform_0_10.value_counts()
touched = int((w.uniform_0_10 > 0).sum())
out["rdb7_uniform_0_10_swap_mix"] = {"touched": touched, "two_wrong": int(vc.get(2, 0)), "four_wrong": int(vc.get(4, 0)),
                                     "share_of_touched_with_two_swaps": float(vc.get(4, 0) / touched)}
(R / "results" / "analysis" / "operator_counts.json").write_text(json.dumps(out, indent=1))
for ds in ("rdb7", "e2sn2", "cyclo"):
    if ds in out:
        print(ds)
        print(pd.DataFrame(out[ds]).T.round(4).to_string())
print(out["rdb7_uniform_0_10_swap_mix"])
