"""Answer RQ1 to RQ5 from the grid outputs. Every quantity is computed per split, paired on the
same test reactions, then summarised over splits (mean, sd, t interval, hierarchical bootstrap,
paired t test, Holm within dataset). Intervals over splits are given twice: the plain t
interval and the Nadeau and Bengio corrected t interval, which inflates the variance by
(1/J + n_test/n_train) because random splits share training reactions; the corrected one is
primary and its p value enters the Holm family. Each quantity's bootstrap has its own random
stream seeded from (dataset, quantity name), so adding a quantity does not move other intervals.
Quantities that are not differences (MAE levels, shares, crossovers, predictions) carry no p
value; where a hypothesis has a criterion other than zero, a separate "criterion" quantity is
the difference from that criterion. Writes results/analysis/{summary.json, quantities.csv,
curves.csv} and figures. Tolerates missing runs: quantities are computed on the splits available
and the split count is reported with each.

Usage: python3 -I analyse.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[2]
G = R / "results" / "grid"
OUT = R / "results" / "analysis"
LEVELS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # 0.6 and 0.8: RDB7 trained and tested only
NTEST_NTRAIN = 0.125  # 80/10/10 random splits
NDRAW = 3  # corruption draws used everywhere: the operator fractions (x axis) and levels above 0.10 have 3
B_BOOT = 2000
rng = np.random.default_rng(12345)


def seed_rng(*keys):
    global rng
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))


def load(ds):
    """Per reaction predictions of every finished run: dict (tag, split) -> DataFrame."""
    runs = {}
    for d in sorted((G / ds).glob("*_s*")):
        if not (d / "metrics.json").exists():
            continue
        tag, s = d.name.rsplit("_s", 1)
        p = pd.read_csv(d / "test_preds.csv.gz")
        p["ae"] = (p.yhat - p.y).abs()
        runs[(tag, int(s))] = p
    return runs


def err(runs, tag, s, cond):
    """Per reaction absolute error, averaged over corruption draws; Series indexed by id."""
    p = runs.get((tag, s))
    if p is None:
        return None
    q = p[(p.cond == cond) & (p.draw < NDRAW)]
    if q.empty:
        return None
    return q.groupby("id").ae.mean()


def lvl(p):
    return f"within:{p:g}" if p != 1.0 else "within:1.0"


def summarise(name, per_split, boot=None, unit="kcal/mol", test=True):
    """per_split: dict split -> value. test=False for quantities that are not differences."""
    v = np.array([per_split[k] for k in sorted(per_split)], float)
    n = len(v)
    out = dict(quantity=name, n_splits=n, unit=unit)
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    out["sd"] = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * out["sd"] / np.sqrt(n)
        out["ci_t"] = [out["mean"] - h, out["mean"] + h]
        se_nb = out["sd"] * np.sqrt(1 / n + NTEST_NTRAIN)
        h_nb = stats.t.ppf(0.975, n - 1) * se_nb
        out["ci_nb"] = [out["mean"] - h_nb, out["mean"] + h_nb]
        if test:
            out["p_t"] = float(stats.ttest_1samp(v, 0.0).pvalue)
            out["p_nb"] = float(2 * stats.t.sf(abs(out["mean"]) / se_nb, n - 1)) if se_nb > 0 else float("nan")
    if boot is not None:
        out["ci_boot"] = boot
    out["per_split"] = {int(k): float(per_split[k]) for k in sorted(per_split)}
    return out


def fieller(a, b):
    """Ratio of means of paired per split values a/b with a Fieller interval that uses the
    Nadeau and Bengio corrected variance. Returns (ratio, interval or None if unbounded)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    c = 1 / n + NTEST_NTRAIN
    t2 = stats.t.ppf(0.975, n - 1) ** 2
    ma, mb = a.mean(), b.mean()
    saa, sbb, sab = a.var(ddof=1) * c, b.var(ddof=1) * c, np.cov(a, b)[0, 1] * c
    qa, qb, qc = mb ** 2 - t2 * sbb, -2 * (ma * mb - t2 * sab), ma ** 2 - t2 * saa
    disc = qb ** 2 - 4 * qa * qc
    ci = [float((-qb - np.sqrt(disc)) / (2 * qa)), float((-qb + np.sqrt(disc)) / (2 * qa))] if qa > 0 and disc > 0 else None
    return float(ma / mb), ci


def hboot(fn, splits, B=B_BOOT):
    """Hierarchical bootstrap: resample splits, then reactions within split. fn(split, idx_sample)
    returns the per split statistic for a resample of that split's test reactions (idx None =
    all). Returns a 95 percent percentile interval of the mean over splits."""
    splits = list(splits)
    if len(splits) < 2:
        return None
    vals = []
    for _ in range(B):
        ss = rng.choice(splits, len(splits))
        acc = []
        for s in ss:
            acc.append(fn(s, True))
        acc = [a for a in acc if not np.isnan(a)]
        if acc:
            vals.append(np.mean(acc))
    if len(vals) < B / 2:
        return None
    return [float(np.percentile(vals, 2.5, method="nearest")), float(np.percentile(vals, 97.5, method="nearest"))]


class Split:
    """Errors of one split, aligned on the test reaction ids, with a resampling helper."""

    def __init__(self, runs, s):
        self.runs, self.s = runs, s
        base = err(runs, "true", s, "true")
        self.ids = None if base is None else base.index.values

    _cache = {}

    def e(self, tag, cond, sub=None):
        key = (id(self.runs), tag, self.s, cond)
        if key not in Split._cache:
            x = err(self.runs, tag, self.s, cond)
            Split._cache[key] = None if x is None else x.reindex(self.ids)
        x = Split._cache[key]
        if x is None:
            return None
        return x if sub is None else x[sub]


def frac_table(ds):
    f = R / "results" / "operator" / f"{ds}_fractions.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    for suffix in ("extra", "profile"):  # profile: second revision condition, RDB7 only
        x = R / "results" / "operator" / f"{ds}_fractions_{suffix}.csv"
        if x.exists() and x.stat().st_size > 1:
            t = pd.concat([t, pd.read_csv(x)], ignore_index=True)
    t = t[t.draw < NDRAW] if "draw" in t else t
    return t.groupby(["split", "part", "cond"])[["nominal", "effective", "reactions_touched"]].mean()


def xeff(ft, s, part, cond, fallback):
    if ft is not None and (s, part, cond) in ft.index:
        return float(ft.loc[(s, part, cond), "effective"])
    return fallback


def ftval(ft, s, part, cond, col):
    if ft is not None and (s, part, cond) in ft.index:
        return float(ft.loc[(s, part, cond), col])
    return np.nan


def analyse(ds):
    runs = load(ds)
    ft = frac_table(ds)
    res, curves = [], []
    splits_clean = sorted(s for (t, s) in runs if t == "true")
    S = {s: Split(runs, s) for s in splits_clean}

    def per_split_stat(f, splits):
        """f(split_obj, mask or None) -> float or None. Returns dict and bootstrap fn."""
        d = {}
        for s in splits:
            v = f(S[s], None)
            if v is not None and np.isfinite(v):
                d[s] = v

        def bfn(s, _):
            sp = S[s]
            idx = rng.integers(0, len(sp.ids), len(sp.ids))
            v = f(sp, idx)
            return np.nan if v is None else v
        return d, bfn

    def diff(tag_a, cond_a, tag_b="true", cond_b="true", scale=None, sub_fn=None):
        def f(sp, idx):
            a, b = sp.e(tag_a, cond_a), sp.e(tag_b, cond_b)
            if a is None or b is None:
                return None
            m = np.ones(len(a), bool) if sub_fn is None else sub_fn(sp)
            a, b = a.values, b.values
            if idx is not None:
                a, b, m = a[idx], b[idx], m[idx]
            a, b = a[m], b[m]
            v = a.mean() - b.mean()
            return v if scale is None else v / scale(sp)
        return f

    def add(name, f, splits, unit="kcal/mol", test=True):
        seed_rng(ds, name)
        d, bfn = per_split_stat(f, splits)
        res.append(summarise(name, d, hboot(bfn, list(d)) if len(d) > 1 else None, unit, test))
        return d

    clean = {s: S[s].e("true", "true").mean() for s in splits_clean}
    res.append(summarise("clean MAE", clean, test=False))
    rep = add("clean replicate minus clean (retraining null)", diff("truerep", "true"), splits_clean)

    # curves: test only, train only, both, at every level, splits where available
    for p in LEVELS:
        c = lvl(p)
        tag = f"within{p:g}" if p != 1.0 else "within1.0"
        for regime, (ta, ca) in {"test": ("true", c), "train": (tag, "true"), "both": (tag, c)}.items():
            d, bfn = per_split_stat(diff(ta, ca), splits_clean)
            if not d:
                continue
            part = "train" if regime == "train" else "test"
            xs = [xeff(ft, s, part, c, p) for s in d]
            curves.append(dict(dataset=ds, regime=regime, level=p, x_effective=float(np.mean(xs)),
                               delta=float(np.mean(list(d.values()))), sd=float(np.std(list(d.values()), ddof=1)) if len(d) > 1 else np.nan,
                               n=len(d), mae=float(np.mean([clean[s] + d[s] for s in d]))))
    # RQ1 secants at 0.10, per percentage point of heavy atoms (effective)
    sec = {}
    for regime, (ta, ca, part) in {"test": ("true", "within:0.1", "test"), "train": ("within0.1", "true", "train"),
                                   "both": ("within0.1", "within:0.1", "test")}.items():
        sc = (lambda part: (lambda sp: 100 * xeff(ft, sp.s, part, "within:0.1", 0.1)))(part)
        sec[regime] = add(f"secant 0 to 0.10, {regime} regime", diff(ta, ca, scale=sc), splits_clean, "kcal/mol per pp of heavy atoms")

    def h1(sp, idx):
        a = diff("true", "within:0.1", scale=lambda q: 100 * xeff(ft, q.s, "test", "within:0.1", 0.1))(sp, idx)
        b = diff("within0.1", "true", scale=lambda q: 100 * xeff(ft, q.s, "train", "within:0.1", 0.1))(sp, idx)
        return None if a is None or b is None else a - b
    add("H1: test only secant minus train only secant", h1, splits_clean, "kcal/mol per pp")
    a_, b_ = sec["test"], sec["train"]
    common = sorted(set(a_) & set(b_))
    tr_row = next(r for r in res if r["quantity"] == "secant 0 to 0.10, train regime")
    if len(common) > 2 and tr_row.get("ci_nb") and tr_row["ci_nb"][0] > 0:  # plan: ratio only if the train only interval excludes zero
        ratio, ci = fieller([a_[k] for k in common], [b_[k] for k in common])
        res.append(dict(quantity="H1: test only secant over train only secant (ratio of means, corrected Fieller interval)", n_splits=len(common), unit="ratio",
                        mean=ratio, ci_nb=ci, note="Fieller interval unbounded" if ci is None else ""))
    # slopes normalised by reactions touched and by CGR visible wrong atoms per reaction (test only)
    nh = pd.read_csv(R / "data" / f"{ds}.csv").set_index("id").n_heavy
    for p in (0.02, 0.05, 0.1, 0.2, 0.4, 1.0):
        c = lvl(p)
        if p > 0.1:
            add(f"RQ1 test only: MAE increase per touched reaction (kcal/mol on a touched reaction), level {p:g}",
                diff("true", c, scale=lambda sp, c=c: ftval(ft, sp.s, "test", c, "reactions_touched")), splits_clean, "kcal/mol", test=False)
            continue
        add(f"RQ1 test only: MAE increase per pp of reactions touched, level {p:g}",
            diff("true", c, scale=lambda sp, c=c: 100 * ftval(ft, sp.s, "test", c, "reactions_touched")), splits_clean, "kcal/mol per pp of reactions", test=False)
        add(f"RQ1 test only: MAE increase per CGR visible wrong atom per reaction, level {p:g}",
            diff("true", c, scale=lambda sp, c=c: xeff(ft, sp.s, "test", c, p) * nh.reindex(sp.ids).mean()), splits_clean, "kcal/mol per atom", test=False)
        add(f"RQ1 test only: MAE increase per touched reaction (kcal/mol on a touched reaction), level {p:g}",
            diff("true", c, scale=lambda sp, c=c: ftval(ft, sp.s, "test", c, "reactions_touched")), splits_clean, "kcal/mol", test=False)

    # H2 share at 0.10 and 0.20 of the 0 to 1.0 increase, both regime
    for lv, tg in ((0.1, "within0.1"), (0.2, "within0.2")):
        def f(sp, idx, lv=lv, tg=tg):
            a = diff(tg, lvl(lv))(sp, idx)
            b = diff("within1.0", "within:1.0")(sp, idx)
            return None if a is None or b is None or b == 0 else a / b
        add(f"H2: share of the 0 to 1.0 increase reached at {lv:g}, both regime", f, splits_clean, "fraction", test=False)
        def fmax(sp, idx, lv=lv, tg=tg):
            a = diff(tg, lvl(lv))(sp, idx)
            ends = [diff(f"within{q:g}" if q != 1.0 else "within1.0", lvl(q))(sp, idx) for q in LEVELS]
            ends = [e for e in ends if e is not None]
            return None if a is None or len(ends) < 6 or max(ends) <= 0 else a / max(ends)
        add(f"H2: share of the largest increase over the grid reached at {lv:g}, both regime", fmax, splits_clean, "fraction", test=False)
        def g(sp, idx, lv=lv):
            a = diff("true", lvl(lv))(sp, idx)
            b = diff("true", "within:1.0")(sp, idx)
            return None if a is None or b is None or b == 0 else a / b
        add(f"H2: share of the 0 to 1.0 increase reached at {lv:g}, test only regime", g, splits_clean, "fraction", test=False)
        if lv == 0.1:
            def crit(sp, idx, fn):
                v = fn(sp, idx)
                return None if v is None else v - xeff(ft, sp.s, "test", "within:0.1", 0.1) / xeff(ft, sp.s, "test", "within:1.0", 0.87)
            add("H2 criterion: share at 0.1 minus linear null, both regime", lambda sp, idx: crit(sp, idx, f), splits_clean, "fraction")
            add("H2 criterion: share at 0.1 minus linear null, test only regime", lambda sp, idx: crit(sp, idx, g), splits_clean, "fraction")
        else:
            add("H2 criterion: share at 0.2 minus 0.5, both regime", lambda sp, idx: (lambda v: None if v is None else v - 0.5)(f(sp, idx)), splits_clean, "fraction")
    lin = {s: xeff(ft, s, "test", "within:0.1", 0.1) / xeff(ft, s, "test", "within:1.0", 0.87) for s in splits_clean}
    res.append(summarise("H2: linear null share at 0.10 (x(0.10)/x(1.0))", lin, None, "fraction", test=False))
    # monotonicity of the both regime curve: increase at 1.0 minus the largest increase at any level
    def drop(sp, idx):
        ends = [diff(f"within{q:g}" if q != 1.0 else "within1.0", lvl(q))(sp, idx) for q in LEVELS]
        ends = [e for e in ends if e is not None]
        return None if len(ends) < 6 else ends[-1] - max(ends)
    add("H2: both regime increase at 1.0 minus the largest increase over the grid (0 if monotone at the end)", drop, splits_clean, test=False)
    # H3 endpoints
    add("H3: within 1.0 both minus cross trained and tested", diff("within1.0", "within:1.0", "cross", "cross"), splits_clean)
    add("H3: within 1.0 both minus unmapped model", diff("within1.0", "within:1.0", "none", "none"), splits_clean)
    add("cross trained and tested minus clean", diff("cross", "cross"), splits_clean)
    for tg, nm in (("cross", "cross element shuffle trained and tested"), ("none", "unmapped model")):
        d = {s: S[s].e(tg, tg).mean() for s in splits_clean if S[s].e(tg, tg) is not None}
        res.append(summarise(f"{nm}: MAE", d, None, test=False))
    add("unmapped model minus clean", diff("none", "none"), splits_clean)
    add("clean model on cross maps minus clean", diff("true", "cross"), splits_clean)
    add("clean model on unmapped input minus clean", diff("true", "none"), splits_clean)
    # RQ3 pair arms on the same reactions
    def elig_mask(sp):
        """Reactions where a centre, a spectator and a mixed swap all exist."""
        return ELIG[(ds, sp.s)]
    for kind in ("centre", "spect", "mixed"):
        add(f"RQ3: one {kind} swap, MAE increase on eligible reactions", diff("true", f"pair:{kind}", sub_fn=elig_mask), splits_clean)
    add("RQ3: centre swap minus spectator swap", lambda sp, idx: _pdiff(sp, idx, "centre", "spect", elig_mask), splits_clean)
    add("RQ3: mixed swap minus spectator swap", lambda sp, idx: _pdiff(sp, idx, "mixed", "spect", elig_mask), splits_clean)
    pc = {s: diff("true", "pair:centre", sub_fn=elig_mask)(S[s], None) for s in splits_clean}
    ps_ = {s: diff("true", "pair:spect", sub_fn=elig_mask)(S[s], None) for s in splits_clean}
    common = [s for s in splits_clean if pc[s] is not None and ps_[s] is not None]
    if len(common) > 2:
        ratio, ci = fieller([pc[s] for s in common], [ps_[s] for s in common])
        res.append(dict(quantity="RQ3: centre swap increase over spectator swap increase (ratio of means, corrected Fieller interval)", n_splits=len(common),
                        unit="ratio", mean=ratio, ci_nb=ci, note="Fieller interval unbounded" if ci is None else ""))
    def h4(sp, idx):
        c = diff("true", "pair:centre", sub_fn=elig_mask)(sp, idx)
        sv = diff("true", "pair:spect", sub_fn=elig_mask)(sp, idx)
        return None if c is None or sv is None else sv - 0.5 * c
    add("H4 criterion: spectator swap increase minus half the centre swap increase", h4, splits_clean)
    # RQ4 augmentation and fixed copy
    for tag, nm in (("within0.1", "fixed copy 0.10"), ("aug0.1", "redrawn 0.10"), ("aug0.2", "redrawn 0.20"),
                    ("aug0.1rx", "redrawn 0.10, models evaluated on RXNMapper maps")):
        add(f"RQ4 {nm}: clean cost (MAE on clean maps minus clean model)", diff(tag, "true"), splits_clean)
        for q in (0.02, 0.05, 0.1, 0.2, 0.4):
            add(f"RQ4 {nm}: MAE on q={q:g} minus clean model on q={q:g}", diff(tag, lvl(q), "true", lvl(q)), splits_clean)
        def rec(sp, idx, tag=tag):
            a = diff("true", "within:0.1")(sp, idx)
            b = diff(tag, "within:0.1", "true", "within:0.1")(sp, idx)
            return None if a is None or b is None else -b / a
        add(f"RQ4 {nm}: recovery at q=0.10", rec, splits_clean, "fraction")
        if tag == "aug0.1rx":  # second revision: same setting as aug0.1, no criterion rows
            continue
        add(f"H5 criterion {nm}: recovery at q=0.10 minus 0.5", lambda sp, idx, rec=rec: (lambda v: None if v is None else v - 0.5)(rec(sp, idx)), splits_clean, "fraction")
        add(f"H5 criterion {nm}: clean cost minus 0.2", lambda sp, idx, tag=tag: (lambda v: None if v is None else v - 0.2)(diff(tag, "true")(sp, idx)), splits_clean)
    for tag in ("aug0.1",):
        def rdiff(sp, idx, tag=tag):
            a = diff("true", "within:0.1")(sp, idx)
            b1 = diff(tag, "within:0.1", "true", "within:0.1")(sp, idx)
            b2 = diff("within0.1", "within:0.1", "true", "within:0.1")(sp, idx)
            return None if a is None or b1 is None or b2 is None else (-b1 / a) - (-b2 / a)
        add("RQ4 paired: recovery at q=0.10, redrawn 0.10 minus fixed copy 0.10", rdiff, splits_clean, "fraction")
        add("RQ4 paired: clean cost, redrawn 0.10 minus fixed copy 0.10", diff(tag, "true", "within0.1", "true"), splits_clean)
    # RQ5 RXNMapper
    if any(t == "rxnmapper" for (t, s) in runs):
        # first revision: cost per touched reaction of RXNMapper and matched test maps, against the
        # uniform operator (the same quantity at level 0.10 above), clean model
        for c in ("rxnmapper", "matched"):
            add(f"RQ5: clean model on {c} maps, MAE increase per touched reaction (kcal/mol on a touched reaction)",
                diff("true", c, scale=lambda sp, c=c: ftval(ft, sp.s, "test", c, "reactions_touched")), splits_clean, "kcal/mol", test=False)
        # second revision: prediction on the reaction axis, uniform cost per touched reaction at 0.10
        # times RXNMapper's touched share of the test set
        add("RQ5: reaction axis prediction, uniform cost per touched reaction at 0.10 times RXNMapper touched share, test",
            diff("true", "within:0.1", scale=lambda sp: ftval(ft, sp.s, "test", "within:0.1", "reactions_touched") / ftval(ft, sp.s, "test", "rxnmapper", "reactions_touched")),
            splits_clean, test=False)
        # first revision: do models trained on corrupted maps transfer to RXNMapper test maps?
        # fixed copy (within0.1), redrawn (aug0.1rx: the redrawn 0.10 model retrained with the same
        # seeds and evaluated on RXNMapper maps) and the model trained on RXNMapper pairings
        for tag, nm in (("within0.1", "fixed copy 0.10"), ("aug0.1rx", "redrawn 0.10"), ("rxnmapper", "RXNMapper pairings")):
            add(f"RQ4 on RXNMapper test maps, {nm}: MAE minus clean model on the same maps", diff(tag, "rxnmapper", "true", "rxnmapper"), splits_clean)
            def rrec(sp, idx, tag=tag):
                a = diff("true", "rxnmapper")(sp, idx)
                b = diff(tag, "rxnmapper", "true", "rxnmapper")(sp, idx)
                return None if a is None or b is None else -b / a
            add(f"RQ4 on RXNMapper test maps, {nm}: recovery of the clean model's RXNMapper test gap", rrec, splits_clean, "fraction", test=False)
        # second revision: paired comparisons of the three trainings on RXNMapper test maps
        add("RQ4 on RXNMapper test maps, paired: redrawn 0.10 minus fixed copy 0.10 (MAE)", diff("aug0.1rx", "rxnmapper", "within0.1", "rxnmapper"), splits_clean)
        add("RQ4 on RXNMapper test maps, paired: redrawn 0.10 minus RXNMapper pairings (MAE)", diff("aug0.1rx", "rxnmapper", "rxnmapper", "rxnmapper"), splits_clean)
        add("RQ4 on RXNMapper test maps, paired: fixed copy 0.10 minus RXNMapper pairings (MAE)", diff("within0.1", "rxnmapper", "rxnmapper", "rxnmapper"), splits_clean)
        add("RQ4 check: redrawn 0.10 retrained for the RXNMapper test (aug0.1rx) minus redrawn 0.10 (aug0.1), on q=0.10 maps",
            diff("aug0.1rx", "within:0.1", "aug0.1", "within:0.1"), splits_clean)
        add("RQ5: clean model on RXNMapper maps minus clean (test only gap)", diff("true", "rxnmapper"), splits_clean)
        add("RQ5: RXNMapper trained and tested minus clean (both gap)", diff("rxnmapper", "rxnmapper"), splits_clean)
        add("RQ5: RXNMapper trained, tested on true maps, minus clean (train only gap)", diff("rxnmapper", "true"), splits_clean)
        add("RQ5: clean model on raw RXNMapper strings minus clean", diff("true", "rxnmapper_raw"), splits_clean)
        add("RQ5: clean model on matched maps minus clean", diff("true", "matched"), splits_clean)
        add("RQ5: matched trained and tested minus clean", diff("matched", "matched"), splits_clean)
        def pred(regime, at="rxnmapper"):
            """Uniform curve prediction at the effective rate of condition `at` on the test set.
            Test only: linear interpolation of the clean model's curve through (0, 0), 0.02, 0.05,
            0.10 and 0.20 (0.20 added in the first revision: on some splits the RXNMapper and matched
            rates exceed x at 0.10, where np.interp would clamp). Both regime: the secant from 0
            to 0.10 (the only trained level on every split)."""
            def f(sp, idx):
                xr = xeff(ft, sp.s, "test", at, np.nan)
                if regime == "test":
                    xs, ys = [0.0], [0.0]
                    for q in (0.02, 0.05, 0.1, 0.2):
                        d = diff("true", lvl(q))(sp, idx)
                        if d is None:
                            if q == 0.2:
                                break
                            return None
                        xs.append(xeff(ft, sp.s, "test", lvl(q), q))
                        ys.append(d)
                    return float(np.interp(xr, xs, ys))
                a = diff("within0.1", "within:0.1")(sp, idx)
                return None if a is None else a * xr / xeff(ft, sp.s, "test", "within:0.1", 0.1)
            return f
        for regime, tag, mt in (("test", "true", "true"), ("both", "rxnmapper", "matched")):
            add(f"RQ5: uniform curve prediction at the RXNMapper rate, {regime}", pred(regime), splits_clean, test=False)
            add(f"RQ5: uniform curve prediction at the matched condition's rate, {regime}", pred(regime, "matched"), splits_clean, test=False)
            if ds != "cyclo":  # Cyclo-23-TS: observed gap indistinguishable from zero, differences only (plan)
                def lratio(sp, idx, regime=regime, tag=tag):
                    pr = pred(regime)(sp, idx)
                    ob = diff(tag, "rxnmapper")(sp, idx)
                    return None if pr is None or ob is None or ob <= 0 or pr <= 0 else float(np.log(pr / ob))
                add(f"RQ5: log of predicted over observed RXNMapper gap, {regime} (criterion: interval inside log 0.67 to log 1.5, -0.405 to 0.405)", lratio, splits_clean, "log ratio", test=False)
            def pmo(sp, idx, regime=regime, tag=tag):
                pr = pred(regime)(sp, idx)
                ob = diff(tag, "rxnmapper")(sp, idx)
                return None if pr is None or ob is None else pr - ob
            add(f"RQ5: predicted minus observed RXNMapper gap, {regime}", pmo, splits_clean)
            # three term decomposition of observed minus predicted
            def t_rate(sp, idx, regime=regime, mt=mt):
                a, b = pred(regime, "matched")(sp, idx), pred(regime)(sp, idx)
                return None if a is None or b is None or diff(mt, "matched")(sp, idx) is None else a - b
            add(f"RQ5 decomposition, {regime}: rate term, uniform at matched rate minus uniform at RXNMapper rate", t_rate, splits_clean)
            def t_conc(sp, idx, regime=regime, mt=mt):
                a, b = diff(mt, "matched")(sp, idx), pred(regime, "matched")(sp, idx)
                return None if a is None or b is None else a - b
            add(f"RQ5 decomposition, {regime}: concentration term, matched minus uniform at matched rate", t_conc, splits_clean)
            def rm(sp, idx, regime=regime, tag=tag, mt=mt):
                a = diff(tag, "rxnmapper")(sp, idx)
                b = diff(mt, "matched")(sp, idx)
                return None if a is None or b is None else a - b
            add(f"RQ5 decomposition, {regime}: atom choice term, RXNMapper minus matched", rm, splits_clean)
    # second revision: split of the concentration term with the profile condition, all terms on the
    # retrained clean model (trueprof) that carries its own uniform curve, RXNMapper, matched and
    # profile test maps. observed - predicted = rate + clustering + reaction choice + atom choice.
    if any(t == "trueprof" for (t, s) in runs):
        def predt(at):
            def f(sp, idx):
                xr = xeff(ft, sp.s, "test", at, np.nan)
                xs, ys = [0.0], [0.0]
                for q in (0.02, 0.05, 0.1, 0.2):
                    d = diff("trueprof", lvl(q), "trueprof", "true")(sp, idx)
                    if d is None:
                        return None
                    xs.append(xeff(ft, sp.s, "test", lvl(q), q))
                    ys.append(d)
                return float(np.interp(xr, xs, ys))
            return f
        g = {c: diff("trueprof", c, "trueprof", "true") for c in ("rxnmapper", "matched", "profile")}
        add("RQ5 profile model: clean model (retrained) on RXNMapper maps minus clean", g["rxnmapper"], splits_clean)
        add("RQ5 profile model: clean model (retrained) on matched maps minus clean", g["matched"], splits_clean)
        add("RQ5 profile model: clean model (retrained) on profile maps minus clean", g["profile"], splits_clean)
        add("RQ5 profile model: uniform prediction at the RXNMapper rate", predt("rxnmapper"), splits_clean, test=False)
        add("RQ5 profile model: uniform prediction at the profile rate", predt("profile"), splits_clean, test=False)
        def term(fa, fb):
            def f(sp, idx):
                a_, b_ = fa(sp, idx), fb(sp, idx)
                return None if a_ is None or b_ is None else a_ - b_
            return f
        add("RQ5 split, test: rate term, uniform at profile rate minus uniform at RXNMapper rate", term(predt("profile"), predt("rxnmapper")), splits_clean)
        add("RQ5 split, test: clustering term, profile minus uniform at profile rate", term(g["profile"], predt("profile")), splits_clean)
        add("RQ5 split, test: reaction choice term, matched minus profile", term(g["matched"], g["profile"]), splits_clean)
        add("RQ5 split, test: atom choice term, RXNMapper minus matched", term(g["rxnmapper"], g["matched"]), splits_clean)
        add("RQ5 split, test: predicted minus observed", term(predt("rxnmapper"), g["rxnmapper"]), splits_clean)
        for c in ("profile", "matched", "rxnmapper"):
            add(f"RQ5 profile model: MAE increase per touched reaction, {c} maps (kcal/mol on a touched reaction)",
                diff("trueprof", c, "trueprof", "true", scale=lambda sp, c=c: ftval(ft, sp.s, "test", c, "reactions_touched")), splits_clean, "kcal/mol", test=False)
        add("RQ5 profile model: MAE increase per touched reaction, uniform 0.10 (kcal/mol on a touched reaction)",
            diff("trueprof", "within:0.1", "trueprof", "true", scale=lambda sp: ftval(ft, sp.s, "test", "within:0.1", "reactions_touched")), splits_clean, "kcal/mol", test=False)
    # linearity: quadratic through the origin on levels 0.02, 0.05, 0.10, x in percentage points
    for regime in ("test", "train", "both"):
        def quad(sp, idx, regime=regime, coef=1):
            xs, ys = [], []
            for p in (0.02, 0.05, 0.1):
                tag = f"within{p:g}"
                ta, ca = {"test": ("true", lvl(p)), "train": (tag, "true"), "both": (tag, lvl(p))}[regime]
                d = diff(ta, ca)(sp, idx)
                if d is None:
                    return None
                xs.append(100 * xeff(ft, sp.s, "train" if regime == "train" else "test", lvl(p), p))
                ys.append(d)
            xs = np.array(xs)
            b = np.linalg.lstsq(np.c_[xs, xs ** 2], np.array(ys), rcond=None)[0]
            return float(b[coef])
        add(f"linearity: quadratic coefficient, levels 0 to 0.10, {regime} regime", quad, splits_clean, "kcal/mol per pp squared")
        add(f"linearity: linear coefficient of the same fit, {regime} regime",
            lambda sp, idx, q=quad: q(sp, idx, coef=0), splits_clean, "kcal/mol per pp")
    # RQ2 crossover: effective fraction at which a curve first reaches a map free reference
    def morgan(sp):
        best = None
        for feat in ("concat", "diff"):
            j = R / "results" / "baseline_mfp" / f"{ds}_{feat}_s{sp.s}.json"
            if j.exists():
                v = json.loads(j.read_text())["val_mae"]
                if best is None or v < best[0]:
                    best = (v, feat)
        if best is None:
            return None
        p = pd.read_csv(R / "results" / "baseline_mfp" / f"{ds}_{best[1]}_s{sp.s}_preds.csv.gz").set_index("id")
        return (p.yhat - p.y).abs().reindex(sp.ids)

    def ref_err(sp, ref):
        return sp.e("none", "none") if ref == "unmapped" else morgan(sp)

    def curve_pts(sp, regime):
        """(x effective in pp, per reaction errors) of the curve, from the origin; levels 0.6 and 0.8
        are used where they exist, the six standard levels are required."""
        pts = [(0.0, sp.e("true", "true"))]
        for p in LEVELS:
            tag = "within1.0" if p == 1.0 else f"within{p:g}"
            e = sp.e("true", lvl(p)) if regime == "test" else sp.e(tag, lvl(p))
            if e is None:
                if p in (0.6, 0.8):
                    continue
                return None
            pts.append((100 * xeff(ft, sp.s, "test", lvl(p), p), e))
        return pts

    def cross_at(regime, ref, what="first"):
        """First crossing of the reference by the linearly interpolated curve. NaN if the clean
        model is not better than the reference (no crossover defined), inf if the curve stays below.
        what="recross": 1 if the curve falls below the reference again after the first crossing."""
        def f(sp, idx):
            r = ref_err(sp, ref)
            if r is None or r.isna().any():
                return None
            pts = curve_pts(sp, regime)
            if pts is None:
                return None
            m = (lambda e: e.values.mean()) if idx is None else (lambda e: e.values[idx].mean())
            yr = m(r)
            xy = [(x, m(e)) for x, e in pts]
            if xy[0][1] >= yr:
                return np.nan
            for k, ((x0, y0), (x1, y1)) in enumerate(zip(xy, xy[1:])):
                if y1 >= yr:
                    if what == "recross":
                        return float(any(y < yr for _, y in xy[k + 2:]))
                    return x0 + (yr - y0) * (x1 - x0) / (y1 - y0)
            return 0.0 if what == "recross" else np.inf
        return f
    for regime in ("both", "test"):
        for ref in ("unmapped", "Morgan"):
            f = cross_at(regime, ref)
            raw = {s: f(S[s], None) for s in splits_clean}
            raw = {s: v for s, v in raw.items() if v is not None}
            if not raw:
                continue
            undefined = sum(1 for v in raw.values() if np.isnan(v))
            beyond = sum(1 for v in raw.values() if np.isinf(v))
            rec = {s: cross_at(regime, ref, "recross")(S[s], None) for s in raw}
            add(f"RQ2: crossover with the {ref} reference, {regime} regime (effective pp of heavy atoms)", f, splits_clean, "pp of heavy atoms", test=False)
            res[-1].update(n_beyond_grid=beyond, n_undefined=undefined, n_with_curve=len(raw),
                           n_recross=int(sum(v == 1.0 for v in rec.values() if v is not None and np.isfinite(v))))
            for p in [0.0] + LEVELS:
                def gap(sp, idx, p=p, regime=regime, ref=ref):
                    r = ref_err(sp, ref)
                    if r is None or r.isna().any():
                        return None
                    tag = "within1.0" if p == 1.0 else f"within{p:g}"
                    e = sp.e("true", "true") if p == 0.0 else (sp.e("true", lvl(p)) if regime == "test" else sp.e(tag, lvl(p)))
                    if e is None:
                        return None
                    a, b = e.values, r.values
                    if idx is not None:
                        a, b = a[idx], b[idx]
                    return a.mean() - b.mean()
                add(f"RQ2: {regime} regime curve minus {ref} reference at level {p:g}", gap, splits_clean)
    for ref in ("unmapped", "Morgan"):
        add(f"RQ2: {ref} reference MAE minus clean", lambda sp, idx, ref=ref: (None if ref_err(sp, ref) is None or ref_err(sp, ref).isna().any() else
            (ref_err(sp, ref).values if idx is None else ref_err(sp, ref).values[idx]).mean() - (sp.e("true", "true").values if idx is None else sp.e("true", "true").values[idx]).mean()),
            splits_clean)
    # sensitivity
    add("sensitivity: within 0.10 trained, checkpoint on clean validation, minus clean (train only)", diff("within0.1_valclean", "true"), splits_clean)
    add("sensitivity: within 0.10 trained, checkpoint on clean validation, both regime minus clean", diff("within0.1_valclean", "within:0.1"), splits_clean)
    add("sensitivity 100 epochs: clean MAE minus 50 epoch clean", diff("true_ep100", "true"), splits_clean)
    sp100 = [s for s in splits_clean if ("true_ep100", s) in runs]
    add("sensitivity 50 epochs, same splits as the 100 epoch check: test only gap at 0.10", diff("true", "within:0.1"), sp100)
    add("sensitivity 50 epochs, same splits as the 100 epoch check: both gap at 0.10", diff("within0.1", "within:0.1"), sp100)
    add("sensitivity 50 epochs, same splits as the 100 epoch check: train only gap at 0.10", diff("within0.1", "true"), sp100)
    add("sensitivity 100 epochs: test only gap at 0.10", diff("true_ep100", "within:0.1", "true_ep100", "true"), splits_clean)
    add("sensitivity 100 epochs: both gap at 0.10", diff("within0.1_ep100", "within:0.1", "true_ep100", "true"), splits_clean)
    add("sensitivity 100 epochs: train only gap at 0.10", diff("within0.1_ep100", "true", "true_ep100", "true"), splits_clean)
    # Holm over the headline family
    fam = [r for r in res if r.get("p_nb") is not None and np.isfinite(r["p_nb"]) and r["quantity"].startswith((
        "secant", "H1: test only secant minus", "RQ3: centre swap minus", "RQ3: mixed swap minus", "H2 criterion", "H4 criterion",
        "H5 criterion", "RQ5: clean model on RXNMapper", "RQ5: RXNMapper trained and tested", "RQ5 decomposition, test: concentration",
        "RQ5 decomposition, test: atom choice", "RQ5 decomposition, both: concentration", "RQ5 decomposition, both: atom choice",
        "RQ4 on RXNMapper test maps", "RQ5 split, test: clustering", "RQ5 split, test: reaction choice"))]
    ps = sorted((r["p_nb"], i) for i, r in enumerate(fam))
    m = len(ps)
    adj, run = {}, 0.0
    for j, (p, i) in enumerate(ps):
        run = max(run, min(1.0, (m - j) * p))
        adj[i] = run
    for i, r in enumerate(fam):
        r["p_holm"] = adj[i]
        r["holm_family_size"] = m
    for r in res:
        r["dataset"] = ds
    return res, curves


def _pdiff(sp, idx, a, b, mask_fn):
    ea, eb = sp.e("true", f"pair:{a}"), sp.e("true", f"pair:{b}")
    if ea is None or eb is None:
        return None
    m = mask_fn(sp)
    ea, eb = ea.values, eb.values
    if idx is not None:
        ea, eb, m = ea[idx], eb[idx], m[idx]
    return ea[m].mean() - eb[m].mean()


ELIG = {}


def build_elig(ds, nsp):
    sys.path.insert(0, str(R / "src" / "lib"))
    import amn
    df = pd.read_csv(R / "data" / f"{ds}.csv").set_index("id")
    ok = {i: all(amn.pair_pools(rx).values()) for i, rx in df.rxn.items()}
    sp = json.loads((R / "data" / f"splits_{ds}.json").read_text())
    for s in range(nsp):
        p = G / ds / f"true_s{s}" / "test_preds.csv.gz"
        if not p.exists():
            continue
        ids = pd.read_csv(p).query("cond == 'true'").id.values
        ELIG[(ds, s)] = np.array([ok[i] for i in ids])


def plots(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.DataFrame(curves)
    for ds, g in c.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for regime, col, mk in (("test", "#1f77b4", "o"), ("train", "#2ca02c", "s"), ("both", "#d62728", "^")):
            h = g[g.regime == regime].sort_values("x_effective")
            if h.empty:
                continue
            x = np.r_[0, 100 * h.x_effective.values]
            y = np.r_[0, h.delta.values]
            ax.plot(x, y, marker=mk, color=col, label={"test": "test only", "train": "train only", "both": "both"}[regime], lw=1.5, ms=4)
        ax.set_xlabel("heavy atoms with a wrong partner (%)")
        ax.set_ylabel("MAE increase (kcal/mol)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"curve_{ds}.pdf")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allres, allcurves = [], []
    for ds, nsp in (("rdb7", 10), ("e2sn2", 10), ("cyclo", 7)):
        if not (G / ds).exists():
            continue
        build_elig(ds, nsp)
        r, c = analyse(ds)
        allres += r
        allcurves += c
    (OUT / "summary.json").write_text(json.dumps(allres, indent=1))
    pd.DataFrame(allcurves).to_csv(OUT / "curves.csv", index=False)
    flat = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in allres])
    flat.to_csv(OUT / "quantities.csv", index=False)
    plots(allcurves)
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500)
    print(flat[[c for c in ("dataset", "quantity", "n_splits", "mean", "sd", "ci_t", "p_t", "p_holm") if c in flat]].to_string())


if __name__ == "__main__":
    main()
