"""Build clean heavy atom mapped datasets and fixed splits.

Outputs in research/data/:
  rdb7.csv   id, rxn_explicit, rxn, rxn_rxnmapper, y, n_heavy, n_centre
  e2sn2.csv  id, rxn_explicit, rxn, y, n_heavy, n_centre
  cyclo.csv  id, rxn, rxn_rxnmapper, y, n_heavy, n_centre   (true maps given heavy atom only)
  splits_<ds>.json  {seed: {train: [...], val: [...], test: [...]}} random 80/10/10, seeds 0 to 9
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import amn  # noqa: E402

RAW = HERE.parents[1] / "data" / "raw"
OUT = HERE.parents[1] / "data"


def summarise(df, name):
    print(f"{name}: {len(df)} reactions, heavy atoms mean {df.n_heavy.mean():.2f} "
          f"(min {df.n_heavy.min()}, max {df.n_heavy.max()}), centre atoms mean {df.n_centre.mean():.2f}, "
          f"y mean {df.y.mean():.2f} sd {df.y.std():.2f}")


def build(ids, rxns_explicit, ys, extra=None, already_heavy=False):
    rows = []
    bad = 0
    for i, rx, y in zip(ids, rxns_explicit, ys):
        try:
            h = rx if already_heavy else amn.heavy_rxn(rx)
            n_heavy = len(amn.heavy_atoms(h))
            n_c = len(amn.centre_atoms(h))
        except Exception as e:  # noqa: BLE001
            bad += 1
            print("skip", i, e, file=sys.stderr)
            continue
        row = dict(id=int(i), rxn=h, y=float(y), n_heavy=n_heavy, n_centre=n_c)
        if not already_heavy:
            row["rxn_explicit"] = rx
        if extra is not None:
            row.update(extra[i])
        rows.append(row)
    print("skipped", bad)
    return pd.DataFrame(rows)


def splits(n, name):
    out = {}
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.permutation(n)
        a, b = int(round(0.8 * n)), int(round(0.9 * n))
        out[seed] = dict(train=sorted(idx[:a].tolist()), val=sorted(idx[a:b].tolist()), test=sorted(idx[b:].tolist()))
    (OUT / f"splits_{name}.json").write_text(json.dumps(out))


def main():
    g = pd.read_csv(RAW / "gdb_bench.csv")
    z = pd.read_csv(RAW / "rdb7_zenodo.csv")
    assert (g.idx.values == z.idx.values).all() and np.allclose(g.dE0.values, z.dE0.values)
    extra = {int(r.idx): {"rxn_rxnmapper": r.rxn_smiles_rxnmapper} for r in g.itertuples()}
    rx = (z.rsmi + ">>" + z.psmi).tolist()
    assert rx == g.rxn_smiles.tolist(), "benchmark true maps differ from Zenodo"
    rdb7 = build(z.idx.tolist(), rx, z.dE0.tolist(), extra)
    rdb7.to_csv(OUT / "rdb7.csv", index=False)
    summarise(rdb7, "rdb7")
    splits(len(rdb7), "rdb7")

    e = pd.read_csv(RAW / "e2sn2.csv")
    e2 = build(range(len(e)), e.AAM.tolist(), e.ea.tolist())
    e2.to_csv(OUT / "e2sn2.csv", index=False)
    summarise(e2, "e2sn2")
    splits(len(e2), "e2sn2")

    c = pd.read_csv(RAW / "cyclo_bench.csv")
    extra = {int(r.rxn_id): {"rxn_rxnmapper": r.rxn_smiles_rxnmapper} for r in c.itertuples()}
    cy = build(c.rxn_id.tolist(), c.rxn_smiles.tolist(), c.G_act.tolist(), extra, already_heavy=True)
    cy.to_csv(OUT / "cyclo.csv", index=False)
    summarise(cy, "cyclo")
    splits(len(cy), "cyclo")


if __name__ == "__main__":
    if sys.argv[1:] == ["--splits-only"]:
        for ds in ("rdb7", "e2sn2", "cyclo"):
            splits(len(pd.read_csv(OUT / f"{ds}.csv")), ds)
    else:
        main()
