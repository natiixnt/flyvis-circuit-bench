"""v1.1 analysis Z3 (added after the study, exploratory): agreement between models, variance of learned scalars.

Definitions are those of `docs/analysis-plan-v1.1.md`, section Z3. Nothing here changes a preregistered decision.

* Spearman between the E vectors of every pair of the 10 models; top-61 overlap (count, Jaccard) per pair, the top
  61 being `truth_top_k(E, 61)` (ties at the cut included, zero-effect classes never).
* Per class: mean effect over models, coefficient of variation (sample std, ddof = 1, over mean) across models,
  number of models with the class in the top 61. Classes with mean effect 0 are excluded from the CV and counted.
* Groups: "core" = in the top 61 in at least 9 of 10 models, "variable" = in 1 to 4 of 10, the rest "other".
  For the five-model sets the same fractions are used: core = at least ceil(0.9 n) = 5 of 5, variable = 1 to
  floor(0.4 n) = 2 of 5 (the plan does not spell this out; it is the proportional reading).
* Learned scalar of a class in a model = sum of |weights| (`A`) / synapse count (`S`); its CV across models.
* Spearman between the CV of the scalar and the CV of the effect over classes.
* The ceiling of the leaderboard is a split-half over stimuli inside one model; it is compared with the median
  Spearman between models.

Pairs of models are not independent draws, so pair statistics are given as mean, median, min and max, not as
bootstrap intervals. Quantities per model (split-half) use the study statistics (bootstrap over models, seed
20261101).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..analysis import VERSIONS, ceilings
from ..stats import summarize
from .z1 import DAGGER, threshold_count, truth_top_k

KEY = "z3"
TOP_FRACTION = 0.10
CORE_FRACTION = 0.9
VARIABLE_FRACTION = 0.4


def cv_across_models(x: np.ndarray) -> np.ndarray:
    """Per column CV = sample std / mean over rows (models); nan where the mean is 0."""
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean != 0, sd / mean, np.nan)


def group_bounds(n_models: int) -> tuple[int, int]:
    """(minimum count for core, maximum count for variable) for n_models models."""
    return math.ceil(round(CORE_FRACTION * n_models, 9)), math.floor(round(VARIABLE_FRACTION * n_models, 9))


def groups(n_top: np.ndarray, n_models: int) -> dict[str, np.ndarray]:
    core_min, var_max = group_bounds(n_models)
    core = n_top >= core_min
    variable = (n_top >= 1) & (n_top <= var_max)
    return {"core": core, "variable": variable, "other": ~(core | variable)}


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "n": int(x.size)}


def pair_matrices(e: np.ndarray, k: int) -> dict:
    """Spearman, top-k overlap count and Jaccard between every pair of rows of e (models x classes)."""
    n = e.shape[0]
    top = np.array([truth_top_k(row, k) for row in e])
    sp, inter, jac = (np.eye(n) for _ in range(3))
    inter = np.diag(top.sum(axis=1)).astype(float)
    for i, j in itertools.combinations(range(n), 2):
        sp[i, j] = sp[j, i] = M.spearman(e[i], e[j])
        c = float((top[i] & top[j]).sum())
        u = float((top[i] | top[j]).sum())
        inter[i, j] = inter[j, i] = c
        jac[i, j] = jac[j, i] = c / u if u else float("nan")
    return {"spearman": sp, "overlap": inter, "jaccard": jac, "top": top}


def _pair_summary(mats: dict, idx: list[int], idx2: list[int] | None = None) -> dict:
    """Statistics over pairs inside idx (or between idx and idx2)."""
    if idx2 is None:
        pairs = list(itertools.combinations(idx, 2))
    else:
        pairs = [(i, j) for i in idx for j in idx2]
    return {name: _stats([mats[name][i, j] for i, j in pairs]) for name in ("spearman", "overlap", "jaccard")}


def _spearman_finite(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    return (M.spearman(a[ok], b[ok]) if ok.sum() > 2 else float("nan")), int(ok.sum())


def version_block(e: np.ndarray, scalar: np.ndarray, size: np.ndarray, k: int) -> dict:
    """Per class statistics and groups for one set of models (rows of e and scalar)."""
    n_models = e.shape[0]
    top = np.array([truth_top_k(row, k) for row in e])
    n_top = top.sum(axis=0)
    mean_e = e.mean(axis=0)
    cv_e = cv_across_models(e)
    cv_s = cv_across_models(scalar)
    grp = groups(n_top, n_models)
    rho, n_used = _spearman_finite(cv_s, cv_e)
    core_min, var_max = group_bounds(n_models)
    return {
        "n_models": n_models, "core_min": core_min, "variable_max": var_max,
        "mean_effect": mean_e, "cv_effect": cv_e, "n_in_top": n_top, "cv_scalar": cv_s,
        "mean_scalar": scalar.mean(axis=0),
        "n_mean_zero_excluded": int((mean_e == 0).sum()),
        "group_sizes": {g: int(v.sum()) for g, v in grp.items()},
        "median_cv_effect": {g: float(np.nanmedian(cv_e[v])) if np.isfinite(cv_e[v]).any() else float("nan")
                             for g, v in grp.items()},
        "spearman_cv_scalar_cv_effect": rho, "n_classes_cv_spearman": n_used,
        "groups": grp, "size": size,
    }


def _compact_block(b: dict) -> dict:
    out = {k: v for k, v in b.items() if k not in ("groups", "size", "mean_effect", "cv_effect", "n_in_top",
                                                   "cv_scalar", "mean_scalar")}
    return out


def cross_model_auroc(e: np.ndarray, s: np.ndarray, k: int) -> dict:
    """Added after the Z3 results: does another trained model predict the important classes of model B better than
    anatomy alone? AUROC of E of model A (A != B, all ordered pairs) for the top-k truth of B, next to the AUROC of
    the synapse count (mean over models, equal to every model's own count for the study package) for the same B."""
    n = e.shape[0]
    truth = [truth_top_k(row, k) for row in e]
    anat = [M.auroc(s[b], truth[b]) for b in range(n)]
    other, diff = [], []
    for a, b in itertools.permutations(range(n), 2):
        x = M.auroc(e[a], truth[b])
        other.append(x)
        diff.append(x - anat[b])
    return {"model_a_e": _stats(other), "synapse_count": _stats(anat), "difference": _stats(diff),
            "n_pairs_a_better": int(sum(d > 0 for d in diff)), "n_pairs": len(diff)}


def conclusion(res: dict) -> str:
    """One sentence generated from the numbers."""
    p = res["pairs"]["pooled"]
    g = res["blocks"]["pooled"]["group_sizes"]
    sh = res["split_half_spearman"]["pooled"]["mean"]
    return (f"The median Spearman of E between models is {p['spearman']['median']:.2f} (split-half over stimuli "
            f"inside a model: {sh:.2f}), the median top-{res['k']} overlap is {p['overlap']['median']:g} classes "
            f"(Jaccard {p['jaccard']['median']:.2f}), {g['core']} classes are core and {g['variable']} variable, and "
            f"the Spearman between the CV of the scalar and the CV of the effect is "
            f"{res['blocks']['pooled']['spearman_cv_scalar_cv_effect']:.2f}.")


def analyze(data: dict) -> dict:
    """The `v1_1.z3` block of results.json."""
    models = sorted(ceilings(data))
    n = len(data["models"][models[0]]["B_C1"]["E_B"])
    k = threshold_count(n, TOP_FRACTION)
    e = np.array([data["models"][m]["B_C1"]["E_B"] for m in models], dtype=np.float64)
    a = np.array([data["models"][m]["B_C1"]["A"] for m in models], dtype=np.float64)
    s = np.array([data["models"][m]["B_C1"]["S"] for m in models], dtype=np.float64)
    scalar = a / s
    mats = pair_matrices(e, k)
    idx = {v: [i for i, m in enumerate(models) if m in plan] for v, plan in VERSIONS.items()}
    pairs = {v: _pair_summary(mats, idx[v]) for v in VERSIONS}
    pairs["between_versions"] = _pair_summary(mats, idx["original"], idx["replication"])
    blocks = {v: version_block(e[idx[v]], scalar[idx[v]], s.mean(axis=0), k) for v in VERSIONS}
    stab = {g: int((blocks["original"]["groups"][g] & blocks["replication"]["groups"][g]).sum())
            for g in ("core", "variable")}
    half = {m: M.spearman(np.asarray(data["models"][m]["B_C1"]["E_B1"], dtype=float),
                          np.asarray(data["models"][m]["B_C1"]["E_B2"], dtype=float)) for m in models}
    pooled = blocks["pooled"]
    res = {
        "cross_model_auroc": cross_model_auroc(e, s, k),
        "k": k, "n_classes": n, "models": models,
        "spearman_matrix": mats["spearman"].tolist(), "overlap_matrix": mats["overlap"].tolist(),
        "jaccard_matrix": mats["jaccard"].tolist(),
        "pairs": pairs,
        "split_half_spearman": {v: summarize(half, plan).to_dict() for v, plan in VERSIONS.items()},
        "blocks": {v: _compact_block(b) for v, b in blocks.items()},
        "group_stability_original_vs_replication": stab,
        "classes": {
            "mean_effect": pooled["mean_effect"].tolist(), "cv_effect": pooled["cv_effect"].tolist(),
            "n_in_top": pooled["n_in_top"].tolist(), "mean_scalar": pooled["mean_scalar"].tolist(),
            "cv_scalar": pooled["cv_scalar"].tolist(), "synapses": s.mean(axis=0).tolist(),
            "core": np.flatnonzero(pooled["groups"]["core"]).tolist(),
            "variable": np.flatnonzero(pooled["groups"]["variable"]).tolist(),
        },
    }
    res["conclusion"] = conclusion(res)
    return res


# ---------------------------------------------------------------- report


def _f(x) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _cell(s: dict) -> str:
    lo, hi = s["ci_bootstrap"]
    return f"{_f(s['mean'])} [{_f(lo)}, {_f(hi)}]"


def _pair_row(label: str, p: dict) -> str:
    def g(m: str) -> str:
        fmt = (lambda x: f"{x:g}") if m == "overlap" else _f
        return f"{fmt(p[m]['median'])} ({fmt(p[m]['min'])} to {fmt(p[m]['max'])})"

    c = [g(m) for m in ("spearman", "overlap", "jaccard")]
    return f"| {label} {DAGGER} | {p['spearman']['n']} | " + " | ".join(c) + " |"


def _matrix(label: str, mat, models, fmt) -> list[str]:
    L = [f"### {label} {DAGGER}", "", "| model | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for m, row in zip(models, mat, strict=True):
        L.append(f"| {m} | " + " | ".join(fmt(x) for x in row) + " |")
    return L + [""]


def render_markdown(r: dict) -> str:
    z = r["v1_1"]["z3"]
    models = z["models"]
    k = z["k"]
    sh, p = z["split_half_spearman"]["pooled"], z["pairs"]["pooled"]
    L = [f"# Z3 {DAGGER}: agreement between models, variance of learned scalars", "",
         f"{DAGGER} Added after the study, exploratory, defined in `analysis-plan-v1.1.md` before it was run. "
         "It changes no preregistered decision. Pair statistics are over pairs of models (not independent, so no "
         "interval); the split-half uses the mean over models with a 95% bootstrap interval over models, seed "
         "20261101.", "",
         f"**Conclusion {DAGGER}.** {z['conclusion']}", "",
         f"## Agreement between models {DAGGER}", "",
         f"Top {k} = the top 10% of the {z['n_classes']} classes by E in each model (ties at the cut included). "
         "Cells are median (min to max) over pairs.", "",
         "| pairs | n pairs | Spearman of E | top-61 overlap (classes) | Jaccard |", "|---|---|---|---|---|",
         _pair_row("all 10 models", p), _pair_row("discovery (000-004)", z["pairs"]["original"]),
         _pair_row("replication (005-009)", z["pairs"]["replication"]),
         _pair_row("discovery x replication", z["pairs"]["between_versions"]), ""]
    L += _matrix("Spearman of E between models", z["spearman_matrix"], models, _f)
    L += _matrix(f"Top-{k} overlap (classes in common)", z["overlap_matrix"], models, lambda x: f"{x:.0f}")
    L += _matrix(f"Top-{k} Jaccard", z["jaccard_matrix"], models, _f)
    L += [f"## Ceiling and agreement between models {DAGGER}", "",
          "The ceiling of the leaderboard is a split-half over stimuli inside one model (`E_B1` against `E_B2`), "
          "not over models. It therefore measures how well one model's E is reproduced by other stimuli, not "
          "how well it transfers to another trained model.", "",
          f"Mean split-half Spearman: {_cell(sh)} (all 10 models). Median Spearman between models: "
          f"{_f(p['spearman']['median'])} (range {_f(p['spearman']['min'])} to {_f(p['spearman']['max'])}, "
          f"{p['spearman']['n']} pairs). The between-model median is "
          f"{'below' if p['spearman']['median'] < sh['mean'] else 'not below'} the split-half value.", "",
          f"## Per class variation {DAGGER}", "",
          "CV = sample standard deviation over the mean across models; classes with mean effect 0 are excluded "
          "from the CV and counted. Groups: core = in the top 61 in at least 9 of 10 models, variable = in 1 to 4 "
          "of 10, other = the rest. For five models the same fractions give core = 5 of 5, variable = 1 to 2 of 5. "
          "Clarification of the plan, 2026-11-12: the plan fixes the groups for 10 models only; for the "
          "five-model sets the proportional reading is used because the absolute counts 9 of 10 (core) cannot be "
          "reached with five models.",
          "",
          "| models | n | core | variable | other | mean-0 classes | median CV of E: core | variable | other |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, title in (("original", "discovery (000-004)"), ("replication", "replication (005-009)"),
                     ("pooled", "all 10 models")):
        b = z["blocks"][v]
        g, c = b["group_sizes"], b["median_cv_effect"]
        L.append(f"| {title} {DAGGER} | {b['n_models']} | {g['core']} | {g['variable']} | {g['other']} | "
                 f"{b['n_mean_zero_excluded']} | {_f(c['core'])} | {_f(c['variable'])} | {_f(c['other'])} |")
    st = z["group_stability_original_vs_replication"]
    L += ["", f"Classes that are core in both the discovery and the replication set: {st['core']}; variable in both: "
          f"{st['variable']} {DAGGER}.", "",
          f"## Learned scalar {DAGGER}", "",
          "Scalar of a class in a model = sum of |weights| of the class / synapse count of the class. "
          "Spearman between the CV of the scalar and the CV of the effect across classes:", "",
          "| models | Spearman | classes used |", "|---|---|---|"]
    for v, title in (("original", "discovery (000-004)"), ("replication", "replication (005-009)"),
                     ("pooled", "all 10 models")):
        b = z["blocks"][v]
        L.append(f"| {title} {DAGGER} | {_f(b['spearman_cv_scalar_cv_effect'])} | {b['n_classes_cv_spearman']} |")
    cm = z["cross_model_auroc"]
    L += ["", f"## Does another trained model beat anatomy? {DAGGER} added after Z3 results", "",
          f"{DAGGER} added after Z3 results (2026-11-12), not in the original plan. For every ordered pair of "
          f"different models (A, B) the AUROC of E of model A for the top-{k} truth of model B, next to the AUROC "
          "of the synapse count for the same truth of B. Cells are median (min to max).", "",
          "| score | n | AUROC |", "|---|---|---|",
          f"| E of another trained model {DAGGER} | {cm['model_a_e']['n']} | {_f(cm['model_a_e']['median'])} "
          f"({_f(cm['model_a_e']['min'])} to {_f(cm['model_a_e']['max'])}) |",
          f"| synapse count {DAGGER} | {cm['synapse_count']['n']} | {_f(cm['synapse_count']['median'])} "
          f"({_f(cm['synapse_count']['min'])} to {_f(cm['synapse_count']['max'])}) |",
          f"| difference (other model minus synapse count, same B) {DAGGER} | {cm['difference']['n']} | "
          f"{_f(cm['difference']['median'])} ({_f(cm['difference']['min'])} to {_f(cm['difference']['max'])}) |",
          "", f"The other model is better than the synapse count in {cm['n_pairs_a_better']} of {cm['n_pairs']} "
          f"pairs {DAGGER}.", "",
          f"![scalar against synapse count {DAGGER}](../figures/v1.1-z3-scalar-light.png)", "",
          f"Learned scalar (mean over 10 models) against synapse count; colour = mean ablation effect (log scale), "
          f"the {k} classes with the largest mean effect are ringed {DAGGER}.", ""]
    return "\n".join(L).rstrip("\n") + "\n"


def scalar_figure(results: dict, plt, theme: dict):
    from matplotlib.colors import LogNorm

    from ..figures import WIDTH_IN

    z = results["v1_1"]["z3"]
    c = z["classes"]
    x = np.asarray(c["synapses"], dtype=float)
    y = np.asarray(c["mean_scalar"], dtype=float)
    e = np.asarray(c["mean_effect"], dtype=float)
    pos = e > 0
    top = truth_top_k(e, z["k"])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.2))
    if (~pos).any():
        ax.scatter(x[~pos], y[~pos], s=14, color=theme["muted"], marker="x", lw=0.8, label="mean effect 0")
    sc = ax.scatter(x[pos], y[pos], s=16, c=e[pos], cmap="viridis", norm=LogNorm(e[pos].min(), e[pos].max()),
                    lw=0)
    ax.scatter(x[top], y[top], s=46, facecolors="none", edgecolors=theme["fg"], lw=1.1,
               label=f"top {z['k']} by mean effect")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("synapse count of the class")
    ax.set_ylabel("sum of |weights| / synapse count")
    ax.set_title(f"Learned scalar against synapse count {DAGGER}, 10 models", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.solids.set_rasterized(False)  # vector colorbar: no embedded base64 image in the SVG
    cb.set_label("mean ablation effect")
    ax.legend(loc="upper right")
    return fig


def make_figures(results: dict, out: Path) -> list[Path]:
    from ..figures import THEMES, _mpl, _save

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme_name, theme in THEMES.items():
        plt = _mpl(theme)
        fig = scalar_figure(results, plt, theme)
        paths += _save(fig, out, "v1.1-z3-scalar", theme_name)
        plt.close(fig)
    return paths
