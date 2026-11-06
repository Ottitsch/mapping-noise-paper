"""Unit tests for the corruption operator. Run: python3 -I src/corruption_op/test_amn.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import amn  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"


def elements_by_map(rxn, side):
    s = amn.split(rxn)[side]
    return {a.GetAtomMapNum(): a.GetAtomicNum() for a in amn.mol(s).GetAtoms()}


def test_dataset(name, n=400):
    df = pd.read_csv(DATA / f"{name}.csv").head(n)
    for rid, rx in zip(df.id, df.rxn):
        er, ep = elements_by_map(rx, 0), elements_by_map(rx, 1)
        assert er == ep, "clean map must pair equal elements"
        for p in (0.05, 0.2, 1.0):
            y, S, nh = amn.corrupt(rx, p, np.random.default_rng(rid))
            ep2 = elements_by_map(y, 1)
            assert set(ep2) == set(ep), "map numbers must be a permutation"
            assert all(er[m] == ep2[m] for m in er), "within element: every pairing keeps the element"
            # moved atoms are exactly S, and none is a fixed point
            prod = amn.mol(amn.split(rx)[1]); prod2 = amn.mol(amn.split(y)[1])
            moved = {a.GetAtomMapNum() for a, b in zip(prod.GetAtoms(), prod2.GetAtoms()) if a.GetAtomMapNum() != b.GetAtomMapNum()}
            assert moved == S, (moved, S)
            assert amn.split(y)[0] == amn.split(rx)[0], "reactants untouched"
        y, S, nh = amn.corrupt(rx, 1.0, np.random.default_rng(rid), cross_element=True)
        assert len(S) == nh or nh < 2
        c = amn.centre_atoms(rx)
        y, S, _ = amn.corrupt(rx, 1.0, np.random.default_rng(rid), pool=c)
        assert S <= c, "centre pool respected"
        assert not amn.changed_atoms(rx, rx), "identity changes nothing"
    print(name, "ok")


def test_examples():
    # two methyls on one carbon: swapping them changes no CGR feature
    rx = "[CH3:1][CH:2]([CH3:3])[OH:4]>>[CH3:1][C:2]([CH3:3])=[O:4].[H:5][H:6]"
    rx = "[CH3:1][CH:2]([CH3:3])[OH:4]>>[CH3:1][C:2]([CH3:3])=[O:4]"
    y = amn.relabel_products(rx, {1: 3, 3: 1})
    assert amn.changed_atoms(rx, y) == set(), amn.changed_atoms(rx, y)
    # swapping a methyl carbon with the carbinol carbon changes the graph
    y = amn.relabel_products(rx, {1: 2, 2: 1})
    assert {1, 2} <= amn.changed_atoms(rx, y)
    assert amn.centre_atoms(rx) == {2, 4}, amn.centre_atoms(rx)
    # heavy atom stripping keeps bare H explicit and balanced
    h = amn.heavy_rxn("[H:1][H:2].[C:3]([H:4])([H:5])=[C:6]([H:7])[H:8]>>[C:3]([H:1])([H:4])([H:5])[C:6]([H:2])([H:7])[H:8]")
    assert "[H:1]" in h and "[H:2]" in h, h
    print("examples ok")


if __name__ == "__main__":
    test_examples()
    for ds in ("rdb7", "e2sn2", "cyclo"):
        test_dataset(ds)
