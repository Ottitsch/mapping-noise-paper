"""Quantities added in the second revision.

1. Realised nominal and effective fractions and touched share of the `profile` test condition on
   RDB7 splits 0 to 7, draws 0 to 2 (same salts as the training driver), written to
   results/operator/rdb7_fractions_profile.csv (read by analyse.py with the other fraction files).
2. Cost per touched reaction on a fixed reaction subset: for the RDB7 clean model of each split,
   the test reactions touched at level 0.10 in a draw are followed through the levels 0.20, 0.40
   and 1.0 of the same draw (corruption is nested across levels), so every level is measured on
   the same reactions. Also the mean number of wrong heavy atoms on that subset per level.
3. The MAE of a mean predictor (training set mean) on each RDB7 test set, for scale.

Usage: python3 -I profile_and_subset.py   (writes results/analysis/profile_and_subset.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "src" / "lib"))
import amn  # noqa: E402
import train  # noqa: E402

DS = "rdb7"
SPLITS = range(8)
DRAWS = range(3)
df = pd.read_csv(R / "data" / f"{DS}.csv")
sp_all = json.loads((R / "data" / f"splits_{DS}.json").read_text())
out = {}


def maps(rx):
    return [a.GetAtomMapNum() for a in amn.mol(amn.split(rx)[1]).GetAtoms() if a.GetAtomicNum() > 1]


def nb_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    se = v.std(ddof=1) * np.sqrt(1 / n + 0.125)
    h = stats.t.ppf(0.975, n - 1) * se
    return [float(v.mean() - h), float(v.mean() + h)]


# 1. profile fractions
rows = []
for s in SPLITS:
    te = df.iloc[sp_all[str(s)]["test"]].reset_index(drop=True)
    for r in DRAWS:
        _, nom, eff = train.apply_condition(te, "profile", f"{DS}|{s}|test|{r}", DS, effective=True)
        rows.append(dict(split=s, part="test", cond="profile", draw=r, nominal=nom, effective=eff,
                         reactions_touched=train.LAST["touched"] / len(te)))
pd.DataFrame(rows).to_csv(R / "results" / "operator" / f"{DS}_fractions_profile.csv", index=False)
out["profile_fractions_mean"] = pd.DataFrame(rows)[["nominal", "effective", "reactions_touched"]].mean().to_dict()
print("profile", out["profile_fractions_mean"], flush=True)

# 2. fixed subset cost per touched reaction
levels = [0.1, 0.2, 0.4, 1.0]
touched_mae, touched_mean_pred = [], []  # clean model at 0.10 and training mean, on the touched reactions
per_split = {p: [] for p in levels}
atoms = {p: [] for p in levels}
for s in SPLITS:
    te = df.iloc[sp_all[str(s)]["test"]].reset_index(drop=True)
    pr = pd.read_csv(R / "results" / "grid" / DS / f"true_s{s}" / "test_preds.csv.gz")
    pr["ae"] = (pr.yhat - pr.y).abs()
    base = pr[pr.cond == "true"].set_index("id").ae
    true_maps = [maps(x) for x in te.rxn]
    inc = {p: [] for p in levels}
    nat = {p: [] for p in levels}
    for r in DRAWS:
        corr = {}
        for p in levels:
            c = f"within:{p:g}" if p != 1.0 else "within:1.0"
            rx, _, _ = train.apply_condition(te, c, f"{DS}|{s}|test|{r}", DS)
            corr[p] = rx
        sub = [i for i, (a, b) in enumerate(zip(te.rxn, corr[0.1])) if a != b]
        ids = te.id.values[sub]
        e10 = pr[(pr.cond == "within:0.1") & (pr.draw == r)].set_index("id").ae.reindex(ids)
        mu = df.iloc[sp_all[str(s)]["train"]].y.mean()
        touched_mae.append(float(e10.mean()))
        touched_mean_pred.append(float((te.y.values[sub] - mu).__abs__().mean()))
        for p in levels:
            c = f"within:{p:g}" if p != 1.0 else "within:1.0"
            e = pr[(pr.cond == c) & (pr.draw == r)].set_index("id").ae
            inc[p].append(float((e.reindex(ids) - base.reindex(ids)).mean()))
            nat[p].append(float(np.mean([sum(x != y for x, y in zip(true_maps[i], maps(corr[p][i]))) for i in sub])))
    for p in levels:
        per_split[p].append(float(np.mean(inc[p])))
        atoms[p].append(float(np.mean(nat[p])))
    print("split", s, {p: round(per_split[p][-1], 2) for p in levels}, flush=True)
out["fixed_subset_cost_per_touched"] = {f"{p:g}": dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), ci_nb=nb_ci(v), per_split=v,
                                                     wrong_atoms_per_reaction=float(np.mean(atoms[p]))) for p, v in per_split.items()}
d = np.array(per_split[1.0]) - np.array(per_split[0.1])
out["fixed_subset_1.0_minus_0.1"] = dict(mean=float(d.mean()), ci_nb=nb_ci(d))

out["touched_at_0.1_clean_model_mae_vs_mean_predictor"] = dict(clean_model_on_corrupted=float(np.mean(touched_mae)), mean_predictor=float(np.mean(touched_mean_pred)), n_split_draws=len(touched_mae))
# profiles placed against available, per split and draw
pl = []
for s in SPLITS:
    te = df.iloc[sp_all[str(s)]["test"]].reset_index(drop=True)
    avail = int((train.rxnmapper_table(DS).loc[te.id].wrong_by_el.fillna("") != "").sum())
    for r in DRAWS:
        placed = sum(1 for x in train.profile_assignment(te, f"{DS}|{s}|test|{r}", DS) if x)
        pl.append(dict(split=s, draw=r, available=avail, placed=placed))
out["profiles_placed"] = pl
# 3. mean predictor
mp = []
for s in SPLITS:
    sp = sp_all[str(s)]
    mu = df.iloc[sp["train"]].y.mean()
    mp.append(float((df.iloc[sp["test"]].y - mu).abs().mean()))
out["mean_predictor_mae"] = dict(mean=float(np.mean(mp)), sd=float(np.std(mp, ddof=1)), per_split=mp)
(R / "results" / "analysis" / "profile_and_subset.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "profile_fractions_mean"}, indent=1)[:3000])
