"""Z2 (v1.1, exploratory †): the ablation truth split by stimulus family.

Definitions are those of `docs/analysis-plan-v1.1.md`, section Z2. Per model the 40 stimuli of set B are split
into the 24 moving edges and the 16 Sintel clips (by the names in `bodzce`); `E_edges(e)` and `E_sintel(e)` are
the means of the per-stimulus relative effects `wzgledne` (604 x 40) over the stimuli of one family. Everything
is computed from the stored per-stimulus effects of `ablation/C1_<model>.npz` (trained networks, condition C1).

Per model: Spearman between the two truths, overlap of their top-61 sets (count, Jaccard), split-half ceiling
within each family (100 random splits of that family's stimuli into halves, seed 20261101: mean Spearman between
the halves, and mean AUROC of one half against the binary truth of the other, the latter is the ceiling used for
the share of signal), and the whole method ladder (AUROC, share of signal) against each family's truth. A
method gives one score per class that does not depend on the family, so the same scores are used for both.
Nothing is selected, so the three versions (all 10, discovery 000-004, replication 005-009) differ only by
the models they average. Statistics are those of the study (`stats.summarize`).
"""

from __future__ import annotations

import numpy as np

from .. import metrics as M
from ..analysis import LEADERBOARD, VERSIONS, c1_scores
from ..stats import SEED, summarize

KEY = "z2"
FAMILIES = {"edges": "krawedz", "sintel": "sintel"}
N_SPLITS = 100
TOP_K_FRACTION = M.TOP_FRACTION


def family_index(names: np.ndarray) -> dict[str, np.ndarray]:
    """Indices (into the stimulus axis) of each family; a stimulus that fits no family is an error."""
    names = np.asarray(names).astype(str)
    idx = {fam: np.flatnonzero([n.startswith(prefix) for n in names]) for fam, prefix in FAMILIES.items()}
    if sum(len(v) for v in idx.values()) != len(names):
        raise ValueError("some stimuli belong to neither moving edges nor Sintel")
    return idx


def family_effects(relative: np.ndarray, names: np.ndarray) -> dict[str, np.ndarray]:
    """E per family: mean over that family's stimuli of the per-stimulus relative effect, shape (classes,)."""
    relative = np.asarray(relative, dtype=np.float64)
    return {fam: relative[:, ix].mean(axis=1) for fam, ix in family_index(names).items()}


def split_half(relative_family: np.ndarray, model: str, family: str, n_splits: int = N_SPLITS) -> dict:
    """Ceiling of one family: random halves of the stimuli, mean Spearman and mean AUROC of half a against the
    binary truth of half b. The generator is seeded by (seed, model, family), so the order of the work is free."""
    w = np.asarray(relative_family, dtype=np.float64)
    n = w.shape[1]
    rng = np.random.default_rng([SEED, int(model), list(FAMILIES).index(family)])
    rho, auc = [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a = w[:, perm[: n // 2]].mean(axis=1)
        b = w[:, perm[n // 2:]].mean(axis=1)
        rho.append(M.spearman(a, b))
        auc.append(M.auroc(a, M.binary_truth(b)))
    return {"spearman": float(np.mean(rho)), "auroc": float(np.mean(auc))}


def model_result(d: dict, size: np.ndarray | None, model: str) -> dict:
    """Everything of Z2 for one model (`d` = the files of the model, as in `data.load`)."""
    b = d["B_C1"]
    names = b["bodzce"]
    idx = family_index(names)
    rel = np.asarray(b["wzgledne"], dtype=np.float64)
    e = family_effects(rel, names)
    top = {fam: M.binary_truth(e[fam]) for fam in e}
    overlap = int((top["edges"] & top["sintel"]).sum())
    ceil = {fam: split_half(rel[:, idx[fam]], model, fam) for fam in e}
    out = {
        "n_stimuli": {fam: int(len(ix)) for fam, ix in idx.items()},
        "n_positive": {fam: int(t.sum()) for fam, t in top.items()},
        "agreement": {"spearman": M.spearman(e["edges"], e["sintel"]), "top61_overlap": overlap,
                      "top61_jaccard": M.jaccard(top["edges"], top["sintel"])},
        "ceiling_spearman": {fam: c["spearman"] for fam, c in ceil.items()},
        "ceiling_auroc": {fam: c["auroc"] for fam, c in ceil.items()},
        "consistent_with_E_B": bool(np.allclose((e["edges"] * len(idx["edges"]) + e["sintel"]
                                                 * len(idx["sintel"])) / rel.shape[1], b["E_B"], rtol=1e-9)),
    }
    scores = c1_scores(d, size)
    ladder: dict = {}
    for method in (*LEADERBOARD,):
        if method not in scores:
            continue
        row = {}
        for fam in e:
            a = M.auroc(scores[method], top[fam])
            row[fam] = {"AUROC": a, "signal_share": M.signal_share(a, out["ceiling_auroc"][fam])}
        ladder[method] = row
    out["ladder"] = ladder
    return out


def _versions(per_model: dict[str, float]) -> dict:
    return {v: summarize(per_model, plan).to_dict() for v, plan in VERSIONS.items()}


def analyze(data: dict) -> dict:
    """The `v1_1.z2` entry of results.json; `possible: false` when the package has no per-stimulus effects."""
    models = {m: d for m, d in sorted(data["models"].items())
              if "B_C1" in d and "wzgledne" in d["B_C1"] and "bodzce" in d["B_C1"]}
    if not models:
        return {"possible": False, "reason": "per-stimulus effects are not stored in the measurement package"}
    try:
        for d in models.values():
            family_index(d["B_C1"]["bodzce"])
    except ValueError:
        return {"possible": False, "reason": "the stimulus names do not identify moving edges and Sintel clips"}
    st = data.get("structure")
    size = None if st is None else st["n_krawedzi"].astype(np.float64)
    per = {m: model_result(d, size, m) for m, d in models.items()}

    def q(f) -> dict:
        return _versions({m: f(r) for m, r in per.items()})

    fams = tuple(FAMILIES)
    ladder: dict = {}
    for method in LEADERBOARD:
        if not all(method in r["ladder"] for r in per.values()):
            continue
        entry: dict = {}
        for fam in fams:
            entry[fam] = {mi: q(lambda r, mi=mi, fam=fam, method=method: r["ladder"][method][fam][mi])
                          for mi in ("AUROC", "signal_share")}
        entry["AUROC_edges_minus_sintel"] = q(lambda r, method=method: r["ladder"][method]["edges"]["AUROC"]
                                              - r["ladder"][method]["sintel"]["AUROC"])
        ladder[method] = entry
    first = next(iter(per.values()))
    return {
        "possible": True,
        "dagger": True,
        "n_stimuli": first["n_stimuli"],
        "n_splits": N_SPLITS,
        "seed": SEED,
        "consistent_with_E_B": all(r["consistent_with_E_B"] for r in per.values()),
        "n_positive": {fam: sorted({r["n_positive"][fam] for r in per.values()}) for fam in fams},
        "agreement": {k: q(lambda r, k=k: r["agreement"][k]) for k in ("spearman", "top61_overlap",
                                                                     "top61_jaccard")},
        "ceiling_spearman": {fam: q(lambda r, fam=fam: r["ceiling_spearman"][fam]) for fam in fams},
        "ceiling_auroc": {fam: q(lambda r, fam=fam: r["ceiling_auroc"][fam]) for fam in fams},
        "ladder": ladder,
    }


# ---------------------------------------------------------------- markdown and figures

def _f(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _name(method: str) -> str:
    from ..names import METHODS as NAME

    return NAME[method].replace("|", "\\|")


def _cell(s: dict) -> str:
    return f"{_f(s['mean'])} [{_f(s['ci_bootstrap'][0])}, {_f(s['ci_bootstrap'][1])}]"


def render(z2: dict) -> str:
    if not z2.get("possible"):
        return f"# Z2 †: truth split by stimulus family\n\nNot possible: {z2['reason']}.\n"
    n = z2["n_stimuli"]
    lines = [
        "# Z2 †: truth split by stimulus family",
        "",
        "† Exploratory, added after the study; definitions from `docs/analysis-plan-v1.1.md`, written before the",
        "numbers. Trained networks. Values are the mean over models with the 95% bootstrap interval",
        f"over models in brackets. The truth is the mean of the stored per-stimulus effects over the {n['edges']}",
        f"moving edges (`E_edges`) or the {n['sintel']} Sintel clips (`E_sintel`); every method gives one score",
        "per class, so the same score is compared with both truths. Nothing is selected, so the three versions only",
        "differ by the models they average: all 10, discovery (000-004), replication (005-009).",
        "",
    ]
    if not z2["consistent_with_E_B"]:
        lines += ["Warning: the two family means do not reproduce the stored E on set B.", ""]
    lines += ["## Agreement of the two truths †", "",
              "| quantity † | all 10 | discovery 000-004 | replication 005-009 |", "|---|---|---|---|"]
    rows = [("Spearman(E_edges, E_sintel)", z2["agreement"]["spearman"]),
            ("top-61 overlap (classes)", z2["agreement"]["top61_overlap"]),
            ("top-61 Jaccard", z2["agreement"]["top61_jaccard"]),
            ("split-half ceiling, moving edges, Spearman", z2["ceiling_spearman"]["edges"]),
            ("split-half ceiling, Sintel, Spearman", z2["ceiling_spearman"]["sintel"]),
            ("split-half ceiling, moving edges, AUROC", z2["ceiling_auroc"]["edges"]),
            ("split-half ceiling, Sintel, AUROC", z2["ceiling_auroc"]["sintel"])]
    for label, q in rows:
        lines.append(f"| {label} | " + " | ".join(_cell(q[v]) for v in VERSIONS) + " |")
    lines += ["", f"Ceilings: {z2['n_splits']} random splits of the family's stimuli into halves, seed {z2['seed']}",
              "(a half has fewer stimuli than the whole family, so the ceilings are conservative). The AUROC ceiling",
              "is that of half a against the binary truth (top 61) of half b; used for the share of signal.",
              ""]
    for mi, title in (("AUROC", "AUROC"), ("signal_share", "Share of the achievable signal")):
        lines += [f"## {title} against each family's truth †", "",
                  "| method † | moving edges, all 10 | Sintel, all 10 | moving edges, 000-004 | Sintel, 000-004 | "
                  "moving edges, 005-009 | Sintel, 005-009 |", "|---|---|---|---|---|---|---|"]
        for method, entry in z2["ladder"].items():
            cells = [_cell(entry[fam][mi][v]) for v in VERSIONS for fam in FAMILIES]
            lines.append(f"| {_name(method)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Difference of AUROC, moving edges minus Sintel †", "",
              "| method † | all 10 | 000-004 | 005-009 |", "|---|---|---|---|"]
    for method, entry in z2["ladder"].items():
        d = entry["AUROC_edges_minus_sintel"]
        lines.append(f"| {_name(method)} | " + " | ".join(_cell(d[v]) for v in VERSIONS) + " |")
    lines.append("")
    return "\n".join(lines)


def figure_ladder(results: dict, plt, theme: dict):
    """AUROC of each method against the moving-edge truth and the Sintel truth (all 10 models), with the ceilings."""
    from ..figures import MIN_FONT_PT, WIDTH_IN
    from ..names import METHODS as NAME

    z2 = results["v1_1"]["z2"]
    methods = sorted(z2["ladder"], key=lambda k: z2["ladder"][k]["edges"]["AUROC"]["pooled"]["mean"])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.6))
    colors = {"edges": theme["P"], "sintel": theme["S"]}
    for fam, off in (("edges", -0.17), ("sintel", 0.17)):
        for y, k in enumerate(methods):
            s = z2["ladder"][k][fam]["AUROC"]["pooled"]
            ax.plot(s["ci_bootstrap"], [y + off] * 2, color=colors[fam], lw=4, solid_capstyle="butt")
            ax.scatter([s["mean"]], [y + off], color=colors[fam], s=50, zorder=3, edgecolor=theme["bg"])
        c = z2["ceiling_auroc"][fam]["pooled"]["mean"]
        ax.axvline(c, color=colors[fam], lw=1.2, ls=(0, (3, 3)))
    ax.axvline(0.5, color=theme["muted"], lw=1.0)
    ax.set_yticks(range(len(methods)), [NAME[k] for k in methods])
    ax.set_xlabel("AUROC, all 10 models †")
    ax.grid(axis="y", visible=False)
    keys = [plt.Line2D([], [], color=colors[f], lw=4, label=lab) for f, lab in
            (("edges", "moving edges"), ("sintel", "Sintel"))]
    ax.legend(handles=keys, loc="lower right", fontsize=MIN_FONT_PT - 2)
    ax.set_title("Methods against the truth of each stimulus family †", loc="left", fontweight="bold", fontsize=13)
    return fig


def figure_agreement(results: dict, plt, theme: dict):
    """Per model: agreement of the two truths (Spearman, Jaccard of the top 61) and the ceilings of each family."""
    from ..figures import WIDTH_IN

    z2 = results["v1_1"]["z2"]
    rows = [("Spearman of the two truths", z2["agreement"]["spearman"]),
            ("top-61 Jaccard", z2["agreement"]["top61_jaccard"]),
            ("ceiling, moving edges", z2["ceiling_spearman"]["edges"]),
            ("ceiling, Sintel", z2["ceiling_spearman"]["sintel"])]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.6))
    for y, (_, q) in enumerate(rows):
        pm = q["pooled"]["per_model"]
        ax.scatter(list(pm.values()), [y] * len(pm), color=theme["P"], s=36, alpha=0.8, zorder=3)
        ax.scatter([q["pooled"]["mean"]], [y], color=theme["S"], marker="D", s=70, zorder=4)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("per model (dots), mean (diamond) †")
    ax.grid(axis="y", visible=False)
    ax.set_title("Agreement of the two truths, and the ceilings †", loc="left", fontweight="bold", fontsize=13)
    return fig


FIGURES = {"v1.1-z2-ladder": figure_ladder, "v1.1-z2-agreement": figure_agreement}


def render_markdown(results: dict) -> str:
    return render(results["v1_1"][KEY])


def make_figures(results: dict, out) -> list:
    """The two figures, only when the analysis was possible (the package stores per-stimulus effects)."""
    from pathlib import Path

    from ..figures import THEMES, _mpl, _save

    if not results["v1_1"][KEY].get("possible"):
        return []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme_name, theme in THEMES.items():
        plt = _mpl(theme)
        for stem, fn in FIGURES.items():
            fig = fn(results, plt, theme)
            paths += _save(fig, out, stem, theme_name)
            plt.close(fig)
    return paths
