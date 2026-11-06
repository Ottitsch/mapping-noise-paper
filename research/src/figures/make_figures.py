"""Paper figures from the saved results. Run from the repository root:

    python3 -I papers/atom-mapping-error-growth/research/src/figures/make_figures.py [--counts]

Writes paper/figures/{method,dose_response,augmentation,wrong_counts}.pdf and the PNG renders
next to them in research/results/figures/. Curves use splits 0 to 2, the three splits every level
was trained on, and corruption draws 0 to 2, as the analysis does. --counts recomputes the per
reaction wrong atom counts (results/figures/rdb7_wrong_counts.csv), which needs rdkit.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
ROOT = R.parents[2]
G = R / "results" / "grid"
FIG = R.parent / "paper" / "figures"
OUT = R / "results" / "figures"
SPLITS = (0, 1, 2)
NDRAW = 3
plt.style.use(str(ROOT / "style" / "figures.mplstyle"))
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
TRAIN, TEST, BOTH = C[0], C[1], C[2]
GREY = "#8c8c8c"
W = 3.35  # one column, inches


def mae(ds, tag, s, cond):
    f = G / ds / f"{tag}_s{s}" / "test_metrics.csv"
    if not f.exists():
        return np.nan
    t = pd.read_csv(f)
    t = t[(t.cond == cond) & (t.draw < NDRAW)]
    return float(t.mae.mean()) if len(t) else np.nan


def fractions(ds):
    t = pd.read_csv(R / "results" / "operator" / f"{ds}_fractions.csv")
    x = R / "results" / "operator" / f"{ds}_fractions_extra.csv"
    if x.exists():
        t = pd.concat([t, pd.read_csv(x)], ignore_index=True)
    t = t[t.draw < NDRAW]
    return t.groupby(["split", "part", "cond"]).effective.mean()


def lvl(p):
    return "within:1.0" if p == 1.0 else f"within:{p:g}"


def tag(p):
    return "within1.0" if p == 1.0 else f"within{p:g}"


def curve(ds, regime, levels):
    ft = fractions(ds)
    xs, ys, sds = [0.0], [], []
    clean = [mae(ds, "true", s, "true") for s in SPLITS]
    ys.append(np.mean(clean)); sds.append(np.std(clean, ddof=1))
    for p in levels:
        if regime == "test":
            v = [mae(ds, "true", s, lvl(p)) for s in SPLITS]
        elif regime == "train":
            v = [mae(ds, tag(p), s, "true") for s in SPLITS]
        else:
            v = [mae(ds, tag(p), s, lvl(p)) for s in SPLITS]
        if np.isnan(v).any():
            continue
        part = "train" if regime == "train" else "test"
        xs.append(100 * np.mean([ft.loc[(s, part, lvl(p))] for s in SPLITS]))
        ys.append(np.mean(v)); sds.append(np.std(v, ddof=1))
    return np.array(xs), np.array(ys), np.array(sds)


def line(ax, x, y, sd, color, label):
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.18, lw=0)
    ax.plot(x, y, "-o", color=color, label=label, ms=3)


def dose_response():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0))
    rows = []
    for ax, ds, name in zip(axes, ("rdb7", "e2sn2"), ("(a) RDB7", "(b) E2/SN2")):
        levels = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        none = [mae(ds, "none", s, "none") for s in SPLITS]
        # first revision: the no maps reference line is left out (three colours at most); Table 2 holds it
        for regime, color, label in (("test", TEST, "test only"), ("both", BOTH, "both"), ("train", TRAIN, "train only")):
            x, y, sd = curve(ds, regime, levels)
            line(ax, x, y, sd, color, label)
            rows += [dict(dataset=ds, regime=regime, x_effective_pp=a, mae=b, sd=c) for a, b, c in zip(x, y, sd)]
        rows.append(dict(dataset=ds, regime="unmapped", x_effective_pp=np.nan, mae=np.mean(none), sd=np.std(none, ddof=1)))
        ax.set_yscale("log")
        ax.set_title(name, loc="left")
        ax.set_xlim(-2, 90)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_yticks([5, 10, 20, 30])
    axes[1].set_yticks([2.5, 5, 10, 15])
    axes[0].set_ylabel("test MAE, kcal/mol, log scale")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.07), handlelength=1.4, columnspacing=1.0)
    fig.supxlabel("CGR visible wrong heavy atoms, percent", fontsize=7, y=0.02)
    fig.tight_layout(w_pad=0.8)
    save(fig, "dose_response")
    pd.DataFrame(rows).to_csv(OUT / "dose_response.csv", index=False)


def augmentation():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0))
    rows = []
    qs = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    for ax, ds, name in zip(axes, ("rdb7", "e2sn2"), ("(a) RDB7", "(b) E2/SN2")):
        ft = fractions(ds)
        x = [0.0] + [100 * np.mean([ft.loc[(s, "test", lvl(q))] for s in SPLITS]) for q in qs]
        for t, color, label in (("true", TEST, "clean model"), ("within0.1", TRAIN, "fixed copy"),
                                ("aug0.1", BOTH, "redrawn")):
            v = np.array([[mae(ds, t, s, "true")] + [mae(ds, t, s, lvl(q)) for q in qs] for s in SPLITS])
            y, sd = v.mean(0), v.std(0, ddof=1)
            line(ax, np.array(x), y, sd, color, label)
            rows += [dict(dataset=ds, model=t, x_effective_pp=a, mae=b, sd=c) for a, b, c in zip(x, y, sd)]
        ax.set_yscale("log")
        ax.set_title(name, loc="left")
        ax.set_xlim(-2, 90)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_yticks([5, 10, 20, 30])
    axes[1].set_yticks([2.5, 5, 10, 15])
    axes[0].set_ylabel("test MAE, kcal/mol, log scale")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.07), handlelength=1.4, columnspacing=1.0)
    fig.supxlabel("CGR visible wrong heavy atoms in the test maps, percent", fontsize=7, y=0.02)
    fig.tight_layout(w_pad=0.8)
    save(fig, "augmentation")
    pd.DataFrame(rows).to_csv(OUT / "augmentation.csv", index=False)


def wrong_counts_compute():
    sys.path.insert(0, str(R / "src" / "lib"))
    import amn  # noqa: E402
    from train import rng_for  # noqa: E402
    df = pd.read_csv(R / "data" / "rdb7.csv")
    rx = pd.read_csv(R / "results" / "rxnmapper_rate" / "rdb7_per_reaction.csv").set_index("id")
    rows = []
    for rid, r in zip(df.id, df.rxn):
        _, S, _ = amn.corrupt(r, 0.1, rng_for("rdb7|0|test|0", "within", rid))
        rows.append(dict(id=rid, uniform_0_10=len(S), rxnmapper=int(rx.loc[rid, "wrong"])))
    pd.DataFrame(rows).to_csv(OUT / "rdb7_wrong_counts.csv", index=False)


def wrong_counts():
    t = pd.read_csv(OUT / "rdb7_wrong_counts.csv")
    bins = [2, 3, 4]  # second revision: 5 to 7 merged, the 7 bar alone (17 reactions) rendered empty
    lab = ["2", "3", "4", "5+"]
    fig, ax = plt.subplots(figsize=(W, 1.8))
    w = 0.38
    summ = {}
    for i, (col, color, name) in enumerate((("rxnmapper", TEST, "RXNMapper"), ("uniform_0_10", TRAIN, "uniform, 10 percent"))):
        v = t[col].values
        share = [np.mean(v == b) for b in bins] + [np.mean(v >= 5)]
        ax.bar(np.arange(len(lab)) + (i - 0.5) * w, 100 * np.array(share), w, color=color, label=name)
        summ[col] = dict(share_touched=float(np.mean(v > 0)), wrong_atoms_share=float(v.sum() / pd.read_csv(R / "data" / "rdb7.csv").n_heavy.sum()),
                         wrong_per_touched=float(v[v > 0].mean()))
    ax.set_xticks(np.arange(len(lab)), lab)
    ax.set_xlabel("wrong heavy atoms in a reaction")
    ax.set_ylabel("reactions, percent")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "wrong_counts")
    (OUT / "rdb7_wrong_counts_summary.json").write_text(json.dumps(summ, indent=1))


def box(ax, x, y, w, h, text, fc="white", ec="#52514e", size=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, a, b, color="#52514e"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=color, shrinkA=0, shrinkB=0, mutation_scale=6))


def method():
    fig = plt.figure(figsize=(W, 1.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # (a) map as a matching between reactant and product atoms
    ax.text(0.01, 0.96, "(a) within element corruption", fontsize=7, va="top")
    el = ["C", "C", "N", "O"]
    ys = [0.68, 0.53, 0.38, 0.23]
    for k, (x0, title) in enumerate(((0.03, "true map"), (0.30, "corrupted"))):
        ax.text(x0 + 0.09, 0.81, title, ha="center", fontsize=6.8)
        for e, y in zip(el, ys):
            ax.text(x0, y, e, ha="center", va="center", fontsize=7)
            ax.text(x0 + 0.18, y, e, ha="center", va="center", fontsize=7)
        partner = [0, 1, 2, 3] if k == 0 else [1, 0, 2, 3]
        for i, j in enumerate(partner):
            wrong = i != j
            ax.plot([x0 + 0.025, x0 + 0.155], [ys[i], ys[j]], color=TEST if wrong else "#52514e", lw=1.1 if wrong else 0.6)
    arrow(ax, (0.225, 0.49), (0.275, 0.49))
    ax.text(0.255, 0.06, "reactant atoms left, product atoms right", ha="center", fontsize=6.2)
    # (b) regimes
    ax.text(0.58, 0.96, "(b) regimes", fontsize=7, va="top")
    x0, y0, cw, ch = 0.76, 0.20, 0.115, 0.24
    ax.text(x0 + cw, 0.81, "test maps", ha="center", fontsize=6.6)
    ax.text(x0 + cw / 2, 0.71, "true", ha="center", fontsize=6.2)
    ax.text(x0 + 1.5 * cw, 0.71, "wrong, q", ha="center", fontsize=6.2)
    ax.text(0.60, y0 + ch, "train maps", ha="center", va="center", fontsize=6.6, rotation=90)
    ax.text(x0 - 0.008, y0 + 1.5 * ch, "true", ha="right", va="center", fontsize=6.2)
    ax.text(x0 - 0.008, y0 + 0.5 * ch, "wrong, p", ha="right", va="center", fontsize=6.2)
    cells = {(0, 1): ("clean", "white"), (1, 1): ("test\nonly", "#fde3cb"), (0, 0): ("train\nonly", "#d2e3f1"), (1, 0): ("both", "#d5eed5")}
    for (cx, cy), (t, fc) in cells.items():
        box(ax, x0 + cx * cw + 0.004, y0 + cy * ch + 0.008, cw - 0.008, ch - 0.016, t, fc=fc, size=6.2)
    ax.text(0.985, 0.06, "one model per p, tested at every q", ha="right", fontsize=6.2)
    save(fig, "method")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--counts" in sys.argv or not (OUT / "rdb7_wrong_counts.csv").exists():
        wrong_counts_compute()
    method()
    dose_response()
    augmentation()
    wrong_counts()
