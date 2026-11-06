"""Per atom error rate of RXNMapper maps against true heavy atom maps.

For each reaction, the RXNMapper reaction and the true heavy atom reaction describe the same
reactant and product molecules with different atom orders and map numbers. Reactant and product
molecules are aligned by graph matching (all automorphisms, up to a cap), and the alignment that
minimises the number of wrongly paired heavy atoms is kept, so symmetry equivalent assignments count
as correct. The RXNMapper pairing is then written onto the true reaction as a product side
relabelling, which makes it a corruption in the same format as the synthetic operator, and the
same nominal and effective fractions are computed.

Usage: python3 -I rate.py <dataset> <outdir>
"""
import json
import sys
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import amn  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
CAP = 2000


def skeleton(m):
    """Heavy atom skeleton without map numbers; returns (mol, list of original atom indices)."""
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():
        a.SetIntProp("orig", a.GetIdx())
        a.SetAtomMapNum(0)
    for a in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        rw.RemoveAtom(a)
    k = rw.GetMol()
    k.UpdatePropertyCache(strict=False)
    return k, [a.GetIntProp("orig") for a in k.GetAtoms()]


def matches(ref, other):
    """All maps ref heavy atom idx -> other heavy atom idx (original indices) that are
    isomorphisms of the heavy atom skeletons (element and bond type, stereo ignored)."""
    q, qi = skeleton(ref)
    t, ti = skeleton(other)
    if q.GetNumAtoms() != t.GetNumAtoms():
        return []
    ms = t.GetSubstructMatches(q, uniquify=False, useChirality=False, maxMatches=CAP)
    return [{qi[a]: ti[b] for a, b in enumerate(m)} for m in ms]


def analyse(true_rxn, mapper_rxn):
    tr, tp = (amn.mol(s) for s in amn.split(true_rxn))
    mr, mp = (amn.mol(s) for s in amn.split(mapper_rxn))
    mr_ = matches(tr, mr)
    mp_ = matches(tp, mp)
    if not mr_ or not mp_:
        return None
    # true pairing: reactant idx -> product idx in true numbering
    tmap_p = {a.GetAtomMapNum(): a.GetIdx() for a in tp.GetAtoms()}
    T = {a.GetIdx(): tmap_p[a.GetAtomMapNum()] for a in tr.GetAtoms() if a.GetAtomicNum() > 1}
    mmap_p = {a.GetAtomMapNum(): a.GetIdx() for a in mp.GetAtoms() if a.GetAtomMapNum() > 0}
    heavy = [a.GetIdx() for a in tr.GetAtoms() if a.GetAtomicNum() > 1]
    best = None
    for fr in mr_[:200]:
        for fp in mp_[:200]:
            inv_fp = {v: k for k, v in fp.items()}
            M = {}
            for i in heavy:
                mn = mr.GetAtomWithIdx(fr[i]).GetAtomMapNum()
                j = mmap_p.get(mn)
                M[i] = inv_fp[j] if j is not None else None
            wrong = sum(M[i] != T[i] for i in heavy)
            if best is None or wrong < best[0]:
                best = (wrong, M)
            if wrong == 0:
                break
        if best[0] == 0:
            break
    wrong, M = best
    unmapped = sum(M[i] is None for i in heavy)
    # write the RXNMapper pairing onto the true reaction: reactant atom i (true map m_i) must pair
    # with product atom M(i); give product atom M(i) the map number m_i
    newmap = {}
    for i in heavy:
        if M[i] is not None:
            newmap[tp.GetAtomWithIdx(M[i]).GetAtomMapNum()] = tr.GetAtomWithIdx(i).GetAtomMapNum()
    ok_perm = sorted(newmap) == sorted(newmap.values()) and unmapped == 0
    eff = None
    elem_ok = True
    relab, wrong_centre, eff_set, wrong_by_el = None, None, set(), ""
    if ok_perm:
        relab = amn.relabel_products(true_rxn, newmap)
        S = {tr.GetAtomWithIdx(i).GetAtomMapNum() for i in heavy if M[i] != T[i]}
        eff_set = amn.changed_atoms(true_rxn, relab) & S
        eff = len(eff_set)
        wrong_centre = len(S & amn.centre_atoms(true_rxn))
        zc = {}
        for i in heavy:
            if M[i] != T[i]:
                z = tr.GetAtomWithIdx(i).GetAtomicNum()
                zc[z] = zc.get(z, 0) + 1
        wrong_by_el = ";".join(f"{z}:{c}" for z, c in sorted(zc.items()))
        elem_ok = all(tr.GetAtomWithIdx(i).GetAtomicNum() == tp.GetAtomWithIdx(M[i]).GetAtomicNum() for i in heavy)
    return dict(n_heavy=len(heavy), wrong=wrong, effective=eff, wrong_centre=wrong_centre, wrong_by_el=wrong_by_el,
                n_centre=len(amn.centre_atoms(true_rxn)), rxn_relab=relab, unmapped=unmapped, aligned=True,
                perm=ok_perm, within_element=elem_ok, n_align=len(mr_) * len(mp_))


def main():
    ds, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / f"{ds}.csv")
    rows = []
    for r in df.itertuples():
        try:
            res = analyse(r.rxn, r.rxn_rxnmapper)
        except Exception as e:  # noqa: BLE001
            res = None
            print("fail", r.id, e, file=sys.stderr)
        if res is None:
            res = dict(n_heavy=r.n_heavy, aligned=False)
        res["id"] = r.id
        rows.append(res)
    t = pd.DataFrame(rows)
    t.drop(columns=["rxn_relab"]).to_csv(out / f"{ds}_per_reaction.csv", index=False)
    # RXNMapper pairing written onto the true heavy atom reaction, for the training driver
    t[["id", "rxn_relab", "wrong", "effective", "wrong_centre", "wrong_by_el"]].to_csv(DATA / f"{ds}_rxnmapper.csv", index=False)
    a = t[t.aligned & t.perm.fillna(False).astype(bool)]
    rng = np.random.default_rng(0)
    def boot(f):
        idx = np.arange(len(a))
        v = [f(a.iloc[rng.choice(idx, len(idx))]) for _ in range(1000)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    nom = lambda x: x.wrong.sum() / x.n_heavy.sum()
    eff = lambda x: x.effective.sum() / x.n_heavy.sum()
    s = dict(dataset=ds, n=len(t), n_aligned=int(t.aligned.sum()), n_used=len(a),
             n_not_bijective=int((t.aligned & ~t.perm.fillna(False).astype(bool)).sum()),
             cross_element_reactions=int((~a.within_element.astype(bool)).sum()),
             nominal_rate=float(nom(a)), nominal_ci=boot(nom),
             effective_rate=float(eff(a)), effective_ci=boot(eff),
             reactions_with_any_wrong=float((a.wrong > 0).mean()),
             reactions_with_any_effective=float((a.effective > 0).mean()),
             mean_per_reaction_effective=float((a.effective / a.n_heavy).mean()),
             centre_share_of_wrong=float(a.wrong_centre.sum() / a.wrong.sum()),
             centre_share_of_heavy=float(a.n_centre.sum() / a.n_heavy.sum()),
             wrong_per_affected_reaction=float(a.wrong[a.wrong > 0].mean()),
             wrong_count_distribution={int(k): int(v) for k, v in a.wrong.value_counts().sort_index().items()})
    (out / f"{ds}_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
