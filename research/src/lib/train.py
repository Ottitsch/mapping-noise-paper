"""Train one Chemprop v2 CGR model under a given training map condition and evaluate it on test
sets under many map conditions.

Map conditions are strings:
  true                      heavy atom true maps
  within:<p>                within element derangement of a fraction p of heavy atoms
  aug:<p>                   training only: within:<p> redrawn every epoch (augmentation)
  pair:centre|spect|mixed   one swap of two same element atoms, both in the reaction centre, both
                            spectators, or one of each; only in reactions where all three exist
  matched                   within element corruption of exactly as many atoms of each element per
                            reaction as RXNMapper gets wrong on that reaction, at random positions
  profile                   the per reaction RXNMapper error profiles of the test set (wrong atoms per
                            element) moved to randomly chosen test reactions that can host them, at
                            random positions; same clustering as RXNMapper, random reactions
                            (added in the second revision to split the concentration term)
  cross                     cross element derangement of all heavy atoms (control)
  none                      all map numbers removed (control)
  rxnmapper                 RXNMapper pairing written onto the true heavy atom reaction
  rxnmapper_raw             RXNMapper output string as published (different H and stereo writing)
  explicit                  original explicit H maps (reproduction only, parsed with H)
Validation maps follow the training condition unless --val-cond is given. Test conditions are evaluated with R
independent corruption draws each; predictions for every draw are written out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amn  # noqa: E402

DATA = HERE.parents[1] / "data"


def strip_maps(rxn: str) -> str:
    r, p = amn.split(rxn)
    out = []
    for s in (r, p):
        m = amn.mol(s)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        out.append(amn.Chem.MolToSmiles(m))
    return ">>".join(out)


def rng_for(*keys) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


_RXM = {}
LAST = {"touched": 0}  # number of reactions touched by the last stochastic apply_condition call


def rxnmapper_table(name: str) -> pd.DataFrame:
    if name not in _RXM:
        _RXM[name] = pd.read_csv(DATA / f"{name}_rxnmapper.csv").set_index("id")
    return _RXM[name]


def _counts(spec: str) -> dict[int, int]:
    return {int(z): int(c) for z, c in (t.split(":") for t in spec.split(";") if t)}


def profile_assignment(df: pd.DataFrame, salt: str, data_name: str) -> list[str]:
    """Move the non empty RXNMapper error profiles of the reactions in df to random reactions of df
    that have enough heavy atoms of each element. Profiles are placed in random order, each on the
    first free reaction (in a random order of reactions) that can host it; a profile that fits no
    free reaction is dropped. Returns one profile string per reaction ('' = untouched)."""
    rng = rng_for(salt, "profile", "assignment")
    profs = [x for x in rxnmapper_table(data_name).loc[df.id].wrong_by_el.fillna("").tolist() if x]
    profs = [profs[i] for i in rng.permutation(len(profs))]
    comp = []
    for rx in df.rxn:
        c: dict[int, int] = {}
        for _, z in amn.heavy_atoms(rx):
            c[z] = c.get(z, 0) + 1
        comp.append(c)
    order = [int(i) for i in rng.permutation(len(df))]
    out = [""] * len(df)
    for pr in profs:
        need = _counts(pr)
        for j in order:
            if not out[j] and all(comp[j].get(z, 0) >= c for z, c in need.items()):
                out[j] = pr
                break
    return out


def apply_condition(df: pd.DataFrame, cond: str, salt: str, data_name: str = "",
                    effective: bool = False) -> tuple[list[str], float, float]:
    """Return reaction strings under `cond` plus the realised nominal fraction of heavy atoms with
    a wrong partner and, if requested, the effective fraction (wrong atoms whose CGR row or
    incident edges change). Random numbers depend on (salt, family, reaction id) where the family
    is the condition without its level, so corrupted sets are nested across levels."""
    if cond == "true":
        return df.rxn.tolist(), 0.0, 0.0
    if cond == "explicit":
        return df.rxn_explicit.tolist(), 0.0, 0.0
    if cond == "rxnmapper_raw":
        return df.rxn_rxnmapper.tolist(), float("nan"), float("nan")
    if cond == "none":
        return [strip_maps(x) for x in df.rxn], float("nan"), float("nan")
    if cond == "rxnmapper":
        t = rxnmapper_table(data_name).loc[df.id]
        nom = t.wrong.sum() / df.n_heavy.sum()
        return t.rxn_relab.tolist(), float(nom), float(t.effective.sum() / df.n_heavy.sum())
    out, nom, eff, tot = [], 0, 0, 0
    LAST["touched"] = 0
    kind, _, ps = cond.partition(":")
    if kind == "matched":
        byel = rxnmapper_table(data_name).loc[df.id].wrong_by_el.fillna("").tolist()
    if kind == "profile":
        byel = profile_assignment(df, salt, data_name)
    for i, (rid, rx) in enumerate(zip(df.id, df.rxn)):
        rng = rng_for(salt, kind if kind != "pair" else cond, rid)
        if kind == "cross":
            y, S, n = amn.corrupt(rx, 1.0, rng, cross_element=True)
        elif kind == "pair":
            y, S, n = amn.corrupt_pair(rx, ps, rng)
        elif kind in ("matched", "profile"):
            cnt = {int(z): int(c) for z, c in (t.split(":") for t in byel[i].split(";") if t)}
            y, S, n = amn.corrupt_counts(rx, cnt, rng) if cnt else (rx, set(), len(amn.heavy_atoms(rx)))
        elif kind == "within":
            y, S, n = amn.corrupt(rx, float(ps), rng)
        else:
            raise ValueError(cond)
        out.append(y)
        nom += len(S)
        tot += n
        LAST["touched"] += bool(S)
        if S and effective:
            eff += len(amn.changed_atoms(rx, y) & S)
    return out, nom / tot, (eff / tot if effective else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--train-cond", default="true")
    ap.add_argument("--test-conds", default="true")
    ap.add_argument("--draws", type=int, default=3, help="default corruption draws per stochastic test condition; cond@k overrides")
    ap.add_argument("--val-cond", default=None, help="validation map condition (default: the training condition)")
    ap.add_argument("--init-offset", type=int, default=0, help="added to the model seed (replicate null arm)")
    ap.add_argument("--effective", action="store_true", help="also compute effective fractions (slow)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--splits", default=None, help="optional explicit splits json (overrides random splits)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from chemprop import data, featurizers, models, nn

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print("exists, skipping", out)
        return
    t0 = time.time()
    df = pd.read_csv(DATA / f"{a.data}.csv")
    sp = json.loads(Path(a.splits).read_text()) if a.splits else json.loads((DATA / f"splits_{a.data}.json").read_text())[str(a.split_seed)]
    tr, va, te = (df.iloc[sp[k]].reset_index(drop=True) for k in ("train", "val", "test"))
    keep_h = True
    salt_tr = f"{a.data}|{a.split_seed}|train"
    aug = a.train_cond.startswith("aug:")
    base_cond = "within:" + a.train_cond[4:] if aug else a.train_cond
    tr_rx, tr_nom, tr_eff = apply_condition(tr, base_cond, salt_tr + ("|epoch0" if aug else ""), a.data)
    va_rx, _, _ = apply_condition(va, a.val_cond or base_cond, salt_tr + "val", a.data)
    feat = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

    def dset(rxns, ys):
        dps = [data.ReactionDatapoint.from_smi(s, [y], keep_h=keep_h) for s, y in zip(rxns, ys)]
        d = data.ReactionDataset(dps, feat)
        d.cache = True
        return d

    seed = 17 + a.split_seed + a.init_offset
    pl.seed_everything(seed, workers=True)
    dtr, dva = dset(tr_rx, tr.y), dset(va_rx, va.y)
    scaler = dtr.normalize_targets()
    dva.normalize_targets(scaler)
    ltr = data.build_dataloader(dtr, batch_size=a.batch, num_workers=0, seed=seed)
    lva = data.build_dataloader(dva, batch_size=a.batch, num_workers=0, shuffle=False)
    mp = nn.BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=a.hidden, depth=a.depth)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(input_dim=a.hidden, hidden_dim=a.hidden, n_layers=1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.metrics.MAE()],
                        warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)
    ck = ModelCheckpoint(dirpath=out, filename="best", monitor="val_loss", mode="min", save_top_k=1)
    from lightning.pytorch.loggers import CSVLogger

    class DM(pl.LightningDataModule):
        """Training data; with augmentation a new within element corruption of the training maps
        is drawn every epoch (epoch 0 uses the draw built above)."""

        def train_dataloader(self):
            e = self.trainer.current_epoch if self.trainer is not None else 0
            if not aug or e == 0:
                return ltr
            rx, _, _ = apply_condition(tr, base_cond, salt_tr + f"|epoch{e}", a.data)
            d = dset(rx, tr.y)
            d.normalize_targets(scaler)
            return data.build_dataloader(d, batch_size=a.batch, num_workers=0, seed=seed + e)

        def val_dataloader(self):
            return lva

    trainer = pl.Trainer(max_epochs=a.epochs, accelerator="cpu", devices=1,
                         logger=CSVLogger(out, name="log", version=0),
                         enable_progress_bar=False, enable_model_summary=False, callbacks=[ck],
                         deterministic=True, reload_dataloaders_every_n_epochs=1 if aug else 0)
    trainer.fit(model, datamodule=DM())
    t_train = time.time() - t0
    best = models.MPNN.load_from_checkpoint(ck.best_model_path)
    best.eval()

    rows, preds = [], []
    for cspec in a.test_conds.split(","):
        cond, _, k = cspec.partition("@")
        det = cond in ("true", "explicit", "rxnmapper", "rxnmapper_raw", "none")
        for r in range(1 if det else (int(k) if k else a.draws)):
            rx, nom, eff = apply_condition(te, cond, f"{a.data}|{a.split_seed}|test|{r}", a.data, a.effective)
            d = dset(rx, te.y)
            ld = data.build_dataloader(d, batch_size=256, num_workers=0, shuffle=False)
            with torch.inference_mode():
                yhat = torch.cat(trainer.predict(best, ld, return_predictions=True)).numpy().ravel()
            err = np.abs(yhat - te.y.values)
            rows.append(dict(cond=cond, draw=r, mae=float(err.mean()),
                             rmse=float(np.sqrt(((yhat - te.y.values) ** 2).mean())), nominal=nom, effective=eff))
            preds.append(pd.DataFrame(dict(id=te.id, cond=cond, draw=r, y=te.y, yhat=yhat)))
    pd.concat(preds).to_csv(out / "test_preds.csv.gz", index=False, float_format="%.5f")
    import chemprop
    import rdkit
    best_epoch = None
    try:
        lg = pd.read_csv(out / "log" / "version_0" / "metrics.csv")
        v = lg.dropna(subset=["val_loss"])
        best_epoch = int(v.loc[v.val_loss.idxmin(), "epoch"])
    except Exception:  # noqa: BLE001
        pass
    meta = dict(vars(a), train_nominal=tr_nom, train_effective=tr_eff, best_epoch=best_epoch,
                chemprop=chemprop.__version__, rdkit=rdkit.__version__,
                best_val=float(ck.best_model_score), seconds_train=t_train,
                seconds_total=time.time() - t0, n_train=len(tr), n_val=len(va), n_test=len(te),
                torch=torch.__version__, host_arch=os.uname().machine)
    (out / "test_metrics.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
    os.remove(ck.best_model_path) if os.environ.get("AMN_DROP_CKPT") else None
    (out / "metrics.json").write_text(json.dumps(meta, indent=1))
    print(pd.DataFrame(rows).to_string())
    print(json.dumps({k: meta[k] for k in ("best_val", "seconds_train", "seconds_total", "train_nominal", "train_effective")}))


if __name__ == "__main__":
    main()
