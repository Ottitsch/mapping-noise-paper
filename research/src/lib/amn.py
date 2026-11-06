"""Atom map noise utilities: heavy atom maps, within element corruption, effective corruption.

All functions are deterministic given a numpy Generator. Atom maps are handled on
reaction SMILES strings of the form 'reactants>>products' with every atom mapped.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_PARSE = Chem.SmilesParserParams()
_PARSE.removeHs = False


def mol(smi: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smi, _PARSE)
    if m is None:
        raise ValueError(f"bad smiles {smi}")
    return m


def split(rxn: str) -> tuple[str, str]:
    r, a, p = rxn.split(">")
    if a:
        r = f"{r}.{a}"
    return r, p


def _h_keep_set(rm: Chem.Mol, pm: Chem.Mol) -> set[int]:
    """Map numbers of H atoms that must stay explicit: H with no heavy neighbour on either side
    (H2, hydride). Removing them on one side only would leave unmapped atoms."""
    keep = set()
    for m in (rm, pm):
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 1 and not any(n.GetAtomicNum() > 1 for n in a.GetNeighbors()):
                keep.add(a.GetAtomMapNum())
    return keep


def _strip(m: Chem.Mol, keep: set[int]) -> Chem.Mol:
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1 and a.GetAtomMapNum() in keep:
            a.SetIsotope(2)  # protect from removal
    ps = Chem.RemoveHsParameters()
    ps.removeIsotopes = False
    ps.removeMapped = True
    ps.removeHydrides = True
    ps.removeNonimplicit = True
    ps.removeDefiningBondStereo = True
    ps.removeWithWedgedBond = True
    ps.removeDegreeZero = False
    m = Chem.RemoveHs(m, ps, sanitize=True)
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1 and a.GetIsotope() == 2:
            a.SetIsotope(0)
    return m


def heavy_rxn(rxn: str) -> str:
    """Explicit H mapped reaction -> heavy atom mapped reaction with implicit H counts written in
    brackets. H atoms that are bare on one side (H2, H-) stay explicit and mapped."""
    r, p = split(rxn)
    rm, pm = mol(r), mol(p)
    keep = _h_keep_set(rm, pm)
    rs, ps = _strip(rm, keep), _strip(pm, keep)
    out = f"{Chem.MolToSmiles(rs, canonical=False)}>>{Chem.MolToSmiles(ps, canonical=False)}"
    # sanity: map numbers identical on both sides
    a = sorted(x.GetAtomMapNum() for x in rs.GetAtoms())
    b = sorted(x.GetAtomMapNum() for x in ps.GetAtoms())
    if a != b or 0 in a:
        raise ValueError(f"unbalanced after stripping: {rxn}")
    return out


def heavy_atoms(rxn: str) -> list[tuple[int, int]]:
    """(map number, atomic number) of heavy atoms on the product side."""
    _, p = split(rxn)
    return [(a.GetAtomMapNum(), a.GetAtomicNum()) for a in mol(p).GetAtoms() if a.GetAtomicNum() > 1]


def relabel_products(rxn: str, perm: dict[int, int]) -> str:
    """Replace product map numbers m -> perm[m]. Reactants untouched."""
    r, p = split(rxn)
    pm = mol(p)
    for a in pm.GetAtoms():
        m = a.GetAtomMapNum()
        if m in perm:
            a.SetAtomMapNum(int(perm[m]))
    return f"{r}>>{Chem.MolToSmiles(pm, canonical=False)}"


def _derange(rng: np.random.Generator, xs: list[int]) -> list[int]:
    """Uniform random derangement of xs (len >= 2) by rejection."""
    xs = list(xs)
    while True:
        y = [int(v) for v in rng.permutation(xs)]
        if all(a != b for a, b in zip(xs, y)):
            return y


def corrupt(
    rxn: str,
    p: float,
    rng: np.random.Generator,
    pool: set[int] | None = None,
    cross_element: bool = False,
    k_exact: int | None = None,
) -> tuple[str, set[int], int]:
    """Corrupt a fraction p of heavy atoms by a within element derangement of product map numbers.

    The target count k = floor(p * n_heavy + u) with one uniform u per reaction and draw (expected
    value p * n_heavy). Atoms are taken in a random order, fixed per reaction and draw, from `pool`
    (map numbers, default all heavy atoms) restricted to element classes with at least two members
    in the pool. A target of one atom becomes two or zero with probability one half each, since a
    wrong pairing needs two atoms. An element with one selected atom pulls in the next member of
    its class in the order or drops the atom, with probability one half each (drop when no member
    is left). All random numbers are drawn before p is used, so for a fixed rng seed the corrupted
    sets are nested across levels (up to the lone atom rule). k_exact replaces the target count;
    lone atoms then always pull in a partner when one exists. Returns (corrupted rxn, set of heavy
    atom map numbers whose partner changed, number of heavy atoms). With cross_element=True all
    heavy atoms are deranged regardless of element (control only).
    """
    atoms = heavy_atoms(rxn)
    n = len(atoms)
    u_round, u_one = rng.random(), rng.random()
    sub = np.random.default_rng(int(rng.integers(2**62)))
    if n == 0 or (p <= 0 and not k_exact):
        return rxn, set(), n
    if cross_element:
        maps = [m for m, _ in atoms]
        if len(maps) < 2:
            return rxn, set(), n
        y = _derange(sub, maps)
        return relabel_products(rxn, dict(zip(maps, y))), set(maps), n
    el = {m: z for m, z in atoms}
    cand = sorted(m for m, _ in atoms if pool is None or m in pool)
    cls: dict[int, list[int]] = {}
    for m in cand:
        cls.setdefault(el[m], []).append(m)
    cls = {z: ms for z, ms in sorted(cls.items()) if len(ms) >= 2}
    perm_ok = [m for ms in cls.values() for m in ms]
    order = [int(v) for v in sub.permutation(perm_ok)] if perm_ok else []
    coins = sub.random(len(cls))
    if not perm_ok:
        return rxn, set(), n
    if k_exact is not None:
        k = min(k_exact, len(perm_ok))
    else:
        k = min(int(np.floor(p * n + u_round)), len(perm_ok))
        if k == 1:
            k = 2 if u_one < 0.5 else 0
    if k <= 0:
        return rxn, set(), n
    chosen = set(order[:k])
    for coin, (z, ms) in zip(coins, cls.items()):
        sel = [m for m in ms if m in chosen]
        if len(sel) == 1:
            rest = [m for m in order[k:] if el[m] == z and m not in chosen]
            if rest and (k_exact is not None or coin < 0.5):
                chosen.add(rest[0])
            else:
                chosen.discard(sel[0])
    perm = {}
    for z, ms in cls.items():
        sel = [m for m in ms if m in chosen]
        if len(sel) >= 2:
            perm.update(dict(zip(sel, _derange(sub, sel))))
    return relabel_products(rxn, perm), set(perm), n


def corrupt_counts(rxn: str, counts: dict[int, int], rng: np.random.Generator) -> tuple[str, set[int], int]:
    """Derange exactly counts[z] randomly chosen heavy atoms of each element z (counts of 1 are
    impossible for a permutation and do not occur in RXNMapper output)."""
    atoms = heavy_atoms(rxn)
    sub = np.random.default_rng(int(rng.integers(2**62)))
    perm = {}
    for z, c in sorted(counts.items()):
        ms = sorted(m for m, zz in atoms if zz == z)
        if c < 2 or c > len(ms):
            raise ValueError(f"cannot derange {c} of {len(ms)} atoms of element {z}")
        sel = [int(v) for v in sub.choice(ms, size=c, replace=False)]
        perm.update(dict(zip(sel, _derange(sub, sel))))
    return relabel_products(rxn, perm), set(perm), len(atoms)


def pair_pools(rxn: str) -> dict[str, list[tuple[int, int]]]:
    """Candidate single swaps by location: both atoms in the reaction centre, both spectators, or
    one of each (same element in every case)."""
    c = centre_atoms(rxn)
    atoms = heavy_atoms(rxn)
    out = {"centre": [], "spect": [], "mixed": []}
    for i, (a, za) in enumerate(atoms):
        for b, zb in atoms[i + 1:]:
            if za != zb:
                continue
            ka = (a in c) + (b in c)
            out[{2: "centre", 0: "spect", 1: "mixed"}[ka]].append((a, b))
    return out


def corrupt_pair(rxn: str, kind: str, rng: np.random.Generator) -> tuple[str, set[int], int]:
    """Swap exactly one pair of same element atoms of the given location kind, only in reactions
    where all three kinds are possible (so the arms are compared on the same reactions)."""
    pools = pair_pools(rxn)
    n = len(heavy_atoms(rxn))
    j = rng.random()
    if not all(pools.values()):
        return rxn, set(), n
    a, b = pools[kind][int(j * len(pools[kind]))]
    return relabel_products(rxn, {a: b, b: a}), {a, b}, n


# ---------------------------------------------------------------- CGR comparison

_FEAT = None


def featurizer():
    global _FEAT
    if _FEAT is None:
        from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

        _FEAT = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    return _FEAT


def cgr_by_map(rxn: str) -> tuple[dict[int, bytes], dict[tuple[int, int], bytes]]:
    """CGR atom and bond features keyed by reactant map numbers."""
    r, p = split(rxn)
    rm, pm = mol(r), mol(p)
    g = featurizer()((rm, pm))
    mp = [a.GetAtomMapNum() for a in rm.GetAtoms()]
    V = {mp[i]: np.asarray(g.V[i]).tobytes() for i in range(len(mp))}
    E = {}
    for k in range(0, g.edge_index.shape[1], 2):
        u, v = int(g.edge_index[0, k]), int(g.edge_index[1, k])
        if u < len(mp) and v < len(mp):
            key = tuple(sorted((mp[u], mp[v])))
            E[key] = np.asarray(g.E[k]).tobytes()
    return V, E


def changed_atoms(rxn_a: str, rxn_b: str) -> set[int]:
    """Reactant map numbers whose CGR atom row or any incident CGR edge differs between a and b."""
    Va, Ea = cgr_by_map(rxn_a)
    Vb, Eb = cgr_by_map(rxn_b)
    out = {m for m in Va if Va[m] != Vb.get(m)}
    for key in set(Ea) | set(Eb):
        if Ea.get(key) != Eb.get(key):
            out.update(key)
    return out


def centre_atoms(rxn: str) -> set[int]:
    """Map numbers of heavy atoms in the reaction centre under the given map: an atom whose
    own features (H count, charge, degree, hybridisation, aromaticity) change, or that has a bond
    formed, broken or changed in order."""
    r, p = split(rxn)
    rm, pm = mol(r), mol(p)
    pidx = {a.GetAtomMapNum(): a for a in pm.GetAtoms()}
    out = set()

    def desc(a):
        return (a.GetTotalNumHs(), a.GetFormalCharge(), a.GetDegree(), str(a.GetHybridization()), a.GetIsAromatic())

    def bonds(m):
        d = {}
        for b in m.GetBonds():
            k = tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
            d[k] = str(b.GetBondType())
        return d

    for a in rm.GetAtoms():
        if a.GetAtomicNum() > 1 and desc(a) != desc(pidx[a.GetAtomMapNum()]):
            out.add(a.GetAtomMapNum())
    br, bp = bonds(rm), bonds(pm)
    heavy = {a.GetAtomMapNum() for a in rm.GetAtoms() if a.GetAtomicNum() > 1}
    for k in set(br) | set(bp):
        if br.get(k) != bp.get(k):
            out.update(x for x in k if x in heavy)
    return out
