"""Realised nominal and effective corrupted fractions of every test and train map condition, with
the same salts as the training driver, so the numbers describe exactly the maps the models saw.
Also the CGR footprint of single swaps (pair arms): changed CGR atom rows and edges per swap.

Usage: python3 -I operator_table.py <dataset> <n_splits> <outdir> [first_split]
With first_split > 0 only splits first_split to n_splits - 1 are computed, written to
<dataset>_fractions_extra.csv and <dataset>_footprint_extra.csv (the analysis reads both files).
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

DATA = HERE.parents[1] / "data"
LEVELS = ["0.02", "0.05", "0.1", "0.2", "0.4", "1.0"]


def footprint(df, cond, salt, ds):
    """Mean number of CGR atom rows and edges that differ from the clean CGR, per reaction touched."""
    rx, _, _ = apply_condition(df, cond, salt, ds)
    rows = edges = touched = 0
    for a, b in zip(df.rxn, rx):
        if a == b:
            continue
        Va, Ea = amn.cgr_by_map(a)
        Vb, Eb = amn.cgr_by_map(b)
        rows += sum(Va[m] != Vb.get(m) for m in Va)
        edges += sum(Ea.get(k) != Eb.get(k) for k in set(Ea) | set(Eb))
        touched += 1
    return touched, rows / max(touched, 1), edges / max(touched, 1)


def main():
    ds, nsp, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    sfx = "_extra" if first > 0 else ""
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / f"{ds}.csv")
    allsp = json.loads((DATA / f"splits_{ds}.json").read_text())
    rows, fp = [], []
    has_rxm = (DATA / f"{ds}_rxnmapper.csv").exists()
    for s in range(first, nsp):
        sp = allsp[str(s)]
        te = df.iloc[sp["test"]].reset_index(drop=True)
        tr = df.iloc[sp["train"]].reset_index(drop=True)
        conds = [f"within:{p}" for p in LEVELS] + ["cross", "pair:centre", "pair:spect", "pair:mixed"]
        if has_rxm:
            conds += ["matched", "rxnmapper"]
        for c in conds:
            ndraw = 1 if c == "rxnmapper" else 3
            for r in range(ndraw):
                _, nom, eff = apply_condition(te, c, f"{ds}|{s}|test|{r}", ds, effective=True)
                t = train.rxnmapper_table(ds).loc[te.id] if c == "rxnmapper" else None
                touched = float((t.wrong > 0).mean()) if t is not None else train.LAST["touched"] / len(te)
                rows.append(dict(split=s, part="test", cond=c, draw=r, nominal=nom, effective=eff,
                                 reactions_touched=touched))
        for c in [f"within:{p}" for p in LEVELS] + (["matched", "rxnmapper"] if has_rxm else []):
            if s >= 3 and c not in ("within:0.1", "matched", "rxnmapper"):
                continue
            _, nom, eff = apply_condition(tr, c, f"{ds}|{s}|train", ds, effective=True)
            rows.append(dict(split=s, part="train", cond=c, draw=0, nominal=nom, effective=eff))
        if s == 0:
            for c in ["pair:centre", "pair:spect", "pair:mixed", "within:0.1"] + (["rxnmapper", "matched"] if has_rxm else []):
                t, r_, e_ = footprint(te, c, f"{ds}|{s}|test|0", ds)
                fp.append(dict(cond=c, reactions_touched=t, changed_rows_per_reaction=r_, changed_edges_per_reaction=e_))
        print(ds, "split", s, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{ds}_fractions{sfx}.csv", index=False)
    pd.DataFrame(fp).to_csv(out / f"{ds}_footprint{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
