"""Map free baseline: Morgan count fingerprint difference (products minus reactants, radius 2,
2048 bits, maps removed) with histogram gradient boosting. Fixed settings for every dataset; the
number of boosting iterations (up to the cap, 1600 by default, 5000 for RDB7 where the best
iteration sat at the 1600 cap) is the one with the lowest validation MAE.
Two feature sets: "diff" (products minus reactants) and "concat" (reactants next to the
difference). Usage: python3 -I mfp.py <dataset> <split_seed> <outdir> <diff|concat> [max_iter]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import amn  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = amn.mol(smi)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    m = Chem.RemoveHs(m)
    return GEN.GetCountFingerprintAsNumPy(m).astype(np.float32)


def main():
    ds, seed, out, kind = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 1600
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(DATA / f"{ds}.csv")
    F = [(fp(r), fp(p)) for r, p in (amn.split(x) for x in df.rxn)]
    X = np.stack([b - a for a, b in F]) if kind == "diff" else np.stack([np.concatenate([a, b - a]) for a, b in F])
    sp = json.loads((DATA / f"splits_{ds}.json").read_text())[str(seed)]
    tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
    y = df.y.values
    keep = np.abs(X[tr]).sum(0) > 0
    X = X[:, keep]
    m = HistGradientBoostingRegressor(max_iter=cap, learning_rate=0.05, max_leaf_nodes=63,
                                      min_samples_leaf=10, l2_regularization=1.0, max_bins=32,
                                      early_stopping=False, random_state=seed)
    m.fit(X[tr], y[tr])
    vals = [float(np.abs(p - y[va]).mean()) for p in m.staged_predict(X[va])]
    it = int(np.argmin(vals))
    v = vals[it]
    yhat = next(p for i, p in enumerate(m.staged_predict(X[te])) if i == it)
    it += 1
    err = np.abs(yhat - y[te])
    pd.DataFrame(dict(id=df.id.values[te], y=y[te], yhat=yhat)).to_csv(out / f"{ds}_{kind}_s{seed}_preds.csv.gz", index=False, float_format="%.5f")
    res = dict(dataset=ds, seed=seed, features=kind, n_features=int(X.shape[1]), mae=float(err.mean()), rmse=float(np.sqrt(((yhat - y[te]) ** 2).mean())),
               val_mae=v, iterations=it, max_iter=cap, seconds=time.time() - t0)
    (out / f"{ds}_{kind}_s{seed}.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
