"""Translate the five public E2/SN2 folds of Heid and Green into row indices of data/e2sn2.csv."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2]
df = pd.read_csv(R / "data" / "e2sn2.csv")
pos = {s: i for i, s in enumerate(df.rxn_explicit)}
assert len(pos) == len(df), "duplicate reactions"
for k in range(5):
    out = {}
    for part in ("train", "val", "test"):
        f = pd.read_csv(R / "data" / "raw" / "e2sn2_splits" / "e2sn2" / f"fold_{k}" / f"aam_{part}.csv")
        out[part] = [pos[s] for s in f.AAM]
        assert all(abs(df.y[pos[s]] - a) < 1e-9 for s, a in zip(f.AAM, f.act))
    assert len(set(out["train"]) | set(out["val"]) | set(out["test"])) == len(df)
    (R / "data" / f"heidfold_e2sn2_{k}.json").write_text(json.dumps(out))
    print(k, {p: len(v) for p, v in out.items()})
