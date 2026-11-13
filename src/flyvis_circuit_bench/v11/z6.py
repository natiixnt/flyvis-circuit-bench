"""v1.1 analysis Z6 (added after the study, exploratory, optional): silencing cell types against connections.

Definitions are those of `docs/analysis-plan-v1.1.md`, section Z6. Nothing here changes a preregistered decision.

* Silencing the type A = zeroing every class whose source type is A at once (`fcb measure --what silence`), the same
  truth metric as for one class. The plan names 64 types; the 604 classes have 63 distinct source types, the number
  is read from the data.
* Per model: Spearman between the silencing effect of A and the sum, and the maximum, of the effects `E` of the
  classes that leave A.
* "Masking": types whose silencing effect is below 0.5 x the sum of their classes' effects; the reverse: above
  1.5 x the sum. The lists use the mean over models of the silencing effect and of the sum (one ratio per type); for
  every type the number of models in which it is masked, or in the reverse case, is given too. Types whose classes
  have a zero sum are left out and counted.

Statistics as in the study: mean over models, bootstrap and t interval over models, seed 20261101.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..data import MODELS_POOLED
from .z1 import DAGGER, _by_version

KEY = "z6"
MASK_BELOW = 0.5
REVERSE_ABOVE = 1.5


CLARIFICATION = (
    f"**Clarification of the plan, 2026-11-12.** The plan says 64 types. Of the 65 cell types, 63 are the source of "
    "at least one of the 604 classes; the other two have no outgoing class, so there is nothing to silence. "
    f"{DAGGER}")

EXPLODING_FACTOR = 1e3  # a type x model pair counts as exploding when its effect is above this x the median

COST_ESTIMATE = (
    f"**Cost {DAGGER} (measured, 2026-11-12).** 20 s per model on the GPU (RTX 5080, 63 types, 40 stimuli), "
    "about 3.5 min for 10 models. The analytic estimate made before the run (about 74 s per model, from the "
    "ablation step of the study) was 3.7 times too high. For comparison the task measurement of Z4 took 48 s per "
    "model for all 604 classes and used at most 814 MiB of GPU memory.")


def available(data: dict) -> bool:
    return all("SIL" in data["models"].get(m, {}) for m in MODELS_POOLED)


def class_sums(types: np.ndarray, sources: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum and maximum of E over the classes that leave each type (types without classes: 0, 0)."""
    s, mx = np.zeros(len(types)), np.zeros(len(types))
    for i, t in enumerate(types):
        v = np.asarray(e)[np.asarray(sources) == t]
        if v.size:
            s[i], mx[i] = v.sum(), v.max()
    return s, mx


def classify(effect: np.ndarray, total: np.ndarray) -> dict[str, np.ndarray]:
    """Masking (effect < 0.5 x sum), reverse (effect > 1.5 x sum), other; zero sums are excluded (all False)."""
    ok = (total > 0) & np.isfinite(effect)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok, effect / total, np.nan)
    return {"ratio": ratio, "masking": ok & (ratio < MASK_BELOW), "reverse": ok & (ratio > REVERSE_ABOVE),
            "valid": ok}


def per_model(d: dict) -> dict:
    sil = d["SIL"]
    types = np.asarray(sil["tipy"])
    effect = np.asarray(sil["E_B"], dtype=np.float64)
    total, mx = class_sums(types, d["B_C1"]["zrodlo"], np.asarray(d["B_C1"]["E_B"], dtype=np.float64))
    ok = np.isfinite(effect) & (total > 0)
    c = classify(effect, total)
    med = float(np.nanmedian(effect[np.isfinite(effect)]))
    n_exploding = int((effect[np.isfinite(effect)] > EXPLODING_FACTOR * med).sum())
    return {"n_exploding": n_exploding, "median_effect": med, "max_effect": float(np.nanmax(effect)),
            "spearman_sum": M.spearman(effect[ok], total[ok]), "spearman_max": M.spearman(effect[ok], mx[ok]),
            "n_masking": int(c["masking"].sum()), "n_reverse": int(c["reverse"].sum()),
            "n_valid": int(ok.sum()), "n_nonfinite": int((~np.isfinite(effect)).sum()),
            "types": types, "effect": effect, "total": total, "masking": c["masking"], "reverse": c["reverse"]}


def conclusion(res: dict) -> str:
    return (f"Across {res['n_types']} source types the silencing effect has Spearman "
            f"{res['spearman_sum']['pooled']['mean']:.2f} with the sum and "
            f"{res['spearman_max']['pooled']['mean']:.2f} with the maximum of the effects of the outgoing "
            f"classes (mean over models); {len(res['masking_types'])} types are masking (below "
            f"{MASK_BELOW} x the sum) and {len(res['reverse_types'])} the reverse (above {REVERSE_ABOVE} x the sum) "
            f"in the mean over models.")


def analyze(data: dict) -> dict:
    """The `v1_1.z6` block of results.json (needs the silencing measurement of all 10 models)."""
    models = list(MODELS_POOLED)
    per = {m: per_model(data["models"][m]) for m in models}
    types = per[models[0]]["types"]
    if any(not np.array_equal(per[m]["types"], types) for m in models):
        raise ValueError("the type order differs between models")
    eff = np.array([per[m]["effect"] for m in models])
    tot = np.array([per[m]["total"] for m in models])
    n_mask = np.array([per[m]["masking"] for m in models]).sum(axis=0)
    n_rev = np.array([per[m]["reverse"] for m in models]).sum(axis=0)
    mean_eff, mean_tot = np.nanmean(eff, axis=0), tot.mean(axis=0)
    c = classify(mean_eff, mean_tot)
    scalar = {k: _by_version({m: per[m][k] for m in models})
              for k in ("spearman_sum", "spearman_max", "n_masking", "n_reverse", "n_valid", "n_nonfinite")}

    def listing(mask: np.ndarray) -> list[dict]:
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(c["ratio"][idx], kind="stable")]
        return [{"type": str(types[i]), "silencing": float(mean_eff[i]), "sum_of_classes": float(mean_tot[i]),
                 "ratio": float(c["ratio"][i]), "models_masking": int(n_mask[i]), "models_reverse": int(n_rev[i])}
                for i in idx]

    exploding = {m: per[m]["n_exploding"] for m in models}
    res = {"models": models, "n_types": int(len(types)), **scalar,
           "exploding": {"factor": EXPLODING_FACTOR, "pairs": int(sum(exploding.values())),
                         "pairs_total": int(len(types) * len(models)), "per_model": exploding,
                         "max_effect": float(max(per[m]["max_effect"] for m in models))},
           "n_zero_sum_excluded": int((mean_tot <= 0).sum()),
           "masking_types": listing(c["masking"]), "reverse_types": listing(c["reverse"]),
           "scatter": {"type": [str(t) for t in types], "silencing": mean_eff.tolist(),
                       "sum_of_classes": mean_tot.tolist()}}
    res["conclusion"] = conclusion(res)
    return res


# ---------------------------------------------------------------- report


def _f(x) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.3g}"


def _cell(s: dict) -> str:
    lo, hi = s["ci_bootstrap"]
    return f"{_f(s['mean'])} [{_f(lo)}, {_f(hi)}]"


def _row(x: dict) -> str:
    return (f"| `{x['type']}` | {_f(x['silencing'])} | {_f(x['sum_of_classes'])} | {_f(x['ratio'])} | "
            f"{_f(math.log10(x['ratio']))} | {x['models_masking']} | {x['models_reverse']} |")


def exploding_note(z: dict) -> str:
    e = z["exploding"]
    return (f"**Silencing can break the dynamics of the network {DAGGER}.** Silencing a whole type sometimes drives "
            f"the network out of its working range: effects reach {e['max_effect']:.1e}. {e['pairs']} of the "
            f"{e['pairs_total']} type x model pairs have an effect above {e['factor']:g} x the median effect of "
            f"their model. The conclusions therefore rest on Spearman correlations and on ranks, not on the size of "
            f"the effects; the tables of masking and reverse types give the sums and ratios on a log scale (column "
            f"log10 ratio) and are marked {DAGGER}: read them as orders of magnitude.")


def render_markdown(r: dict) -> str:
    z = r["v1_1"][KEY]
    L = [f"# Z6 {DAGGER}: silencing cell types against connections", "",
         f"{DAGGER} Added after the study, exploratory and optional, defined in `analysis-plan-v1.1.md` before it "
         "was run. It changes no preregistered decision. Mean over models with 95% bootstrap interval over models, "
         "seed 20261101.", "",
         "Silencing the type A = zeroing every class whose source type is A at once, same truth metric as for one "
         f"class. {z['n_types']} source types occur among the classes (the plan says 64).", "",
         f"**Conclusion {DAGGER}.** {z['conclusion']}", "", exploding_note(z), "", CLARIFICATION, "", COST_ESTIMATE, "",
         f"## Silencing effect against the effects of the outgoing classes {DAGGER}", "",
         "| quantity | all 10 models | discovery (000-004) | replication (005-009) |", "|---|---|---|---|"]
    for key, label in (("spearman_sum", "Spearman(silencing, sum of class effects)"),
                       ("spearman_max", "Spearman(silencing, maximum class effect)"),
                       ("n_masking", f"types masking (silencing < {MASK_BELOW} x sum)"),
                       ("n_reverse", f"types reverse (silencing > {REVERSE_ABOVE} x sum)"),
                       ("n_valid", "types with a usable ratio"), ("n_nonfinite", "types with a non-finite effect")):
        s = z[key]
        L.append(f"| {label} {DAGGER} | {_cell(s['pooled'])} | {_cell(s['original'])} | {_cell(s['replication'])} |")
    head = [f"| type | silencing {DAGGER} | sum of class effects {DAGGER} | ratio {DAGGER} | log10 ratio {DAGGER} "
            "| models masking | models reverse |", "|---|---|---|---|---|---|---|"]
    L += ["", f"Types left out because the sum of their classes' effects is 0: {z['n_zero_sum_excluded']} "
          f"{DAGGER}.", "",
          f"## Masking types {DAGGER}", "",
          f"Silencing effect below {MASK_BELOW} x the sum (mean over 10 models); {len(z['masking_types'])} types.", "",
          *head, *[_row(x) for x in z["masking_types"]], "",
          f"## Reverse types {DAGGER}", "",
          f"Silencing effect above {REVERSE_ABOVE} x the sum (mean over 10 models); {len(z['reverse_types'])} types.",
          "", *head, *[_row(x) for x in z["reverse_types"]], "",
          "![silencing against the sum of class effects](../figures/v1.1-z6-silencing-light.png)", "",
          f"Silencing effect of every source type against the sum of the effects of its classes (mean over models, "
          f"log scale) {DAGGER}; the lines are y = x, {MASK_BELOW} x and {REVERSE_ABOVE} x.", ""]
    return "\n".join(L).rstrip("\n") + "\n"


def render_pending() -> str:
    """The text of `docs/v1.1/z6.md` until the silencing measurement exists (the tables need it)."""
    return "\n".join([
        f"# Z6 {DAGGER}: silencing cell types against connections", "",
        f"{DAGGER} Added after the study, exploratory and optional, defined in `analysis-plan-v1.1.md` before it "
        "was run. It changes no preregistered decision. The measurement (`fcb measure --what silence`) has not been "
        "run yet; this file is replaced by the tables when it has.", "", CLARIFICATION, "", COST_ESTIMATE, ""])


def silencing_figure(results: dict, plt, theme: dict):
    from ..figures import WIDTH_IN

    z = results["v1_1"][KEY]
    x = np.asarray(z["scatter"]["sum_of_classes"], dtype=float)
    y = np.asarray(z["scatter"]["silencing"], dtype=float)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    ratio = np.where(ok, y / np.where(x > 0, x, 1), np.nan)
    mask, rev = ok & (ratio < MASK_BELOW), ok & (ratio > REVERSE_ABOVE)
    other = ok & ~mask & ~rev
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6))
    ax.scatter(x[other], y[other], s=18, color=theme["muted"], lw=0, label="other types")
    ax.scatter(x[mask], y[mask], s=26, color=theme["P"], lw=0, label=f"masking (< {MASK_BELOW} x sum)")
    ax.scatter(x[rev], y[rev], s=26, color=theme["S"], lw=0, label=f"reverse (> {REVERSE_ABOVE} x sum)")
    lo, hi = float(x[ok].min()), float(x[ok].max())
    xs = np.array([lo, hi])
    for f, ls in ((1.0, "-"), (MASK_BELOW, (0, (3, 3))), (REVERSE_ABOVE, (0, (3, 3)))):
        ax.plot(xs, f * xs, color=theme["muted"], lw=0.8, ls=ls)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sum of the effects of the outgoing classes")
    ax.set_ylabel("silencing effect of the type")
    ax.set_title(f"Silencing a type against its classes {DAGGER}, mean of 10 models", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    return fig


def make_figures(results: dict, out: Path) -> list[Path]:
    from ..figures import THEMES, _mpl, _save

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme_name, theme in THEMES.items():
        plt = _mpl(theme)
        fig = silencing_figure(results, plt, theme)
        paths += _save(fig, out, "v1.1-z6-silencing", theme_name)
        plt.close(fig)
    return paths
