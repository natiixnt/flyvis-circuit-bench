"""v1.1 analysis Z4 (added after the study, exploratory): a second truth at the level of the task.

Definitions are those of `docs/analysis-plan-v1.1.md`, section Z4. Nothing here changes a preregistered decision.

* Task truth `T(e)`: change of the flyvis training loss of the decoded optic flow on the Sintel clips of the study,
  ablated minus intact, mean over clips (measured by `fcb measure --what task`, see `measure/task.py`).
  Binary task truth: the top 61 classes by `T` (ties at the cut included, classes with `T <= 0` never).
* Per model: Spearman(E, T) and the top-61 overlap of E and T (count, Jaccard).
* 2x2 table of classes high or low on E and on T (high = top 61), per model and for the mean over the 10 models,
  with the class lists of the two off-diagonal cells (the lists use the mean over models).
* The full method ladder against `T`: AUROC, AUPRC, precision@61 and the share of signal. The ceiling of `T` is a
  split-half over the Sintel clips (fixed split: even and odd clips; `T` of one half against the top 61 of the
  other), as the ceiling of `E` is a split-half over stimuli (not specified in the plan, stated here).
* Descriptive companion (added after the plan, not a truth): `D(e)` = mean over clips of ||y_e - y_0|| / ||y_0|| of the
  decoded flow; Spearman between `D` and `T` (and with `E`) per model.

Statistics as in the study: mean over models, bootstrap and t interval over models, seed 20261101.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..analysis import LEADERBOARD, c1_scores
from ..data import MODELS_POOLED
from .z1 import DAGGER, _by_version, plain, threshold_count, truth_top_k

KEY = "z4"
TOP_FRACTION = 0.10
METRICS = ("AUROC", "signal_share", "AUPRC", "precision_at_k")


def available(data: dict) -> bool:
    """True when the task measurement exists for every one of the 10 models."""
    return all("T" in data["models"].get(m, {}) for m in MODELS_POOLED)


def two_by_two(e: np.ndarray, t: np.ndarray, k: int) -> dict:
    """Counts of classes high/low on E and on T (high = top k by that quantity)."""
    he, ht = truth_top_k(e, k), truth_top_k(t, k)
    return {"high_E_high_T": int((he & ht).sum()), "high_E_low_T": int((he & ~ht).sum()),
            "low_E_high_T": int((~he & ht).sum()), "low_E_low_T": int((~he & ~ht).sum())}


def _rank_desc(x: np.ndarray) -> np.ndarray:
    """1 = largest."""
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=int)
    r[order] = np.arange(1, len(x) + 1)
    return r


def off_diagonal(e: np.ndarray, t: np.ndarray, k: int, source: np.ndarray, target: np.ndarray) -> dict:
    """Class lists of the two off-diagonal cells, each sorted by the rank on the quantity it is high on."""
    he, ht = truth_top_k(e, k), truth_top_k(t, k)
    re, rt = _rank_desc(e), _rank_desc(t)

    def cell(mask: np.ndarray, by: np.ndarray) -> list[dict]:
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(by[idx], kind="stable")]
        return [{"class": int(i), "source": str(source[i]), "target": str(target[i]), "rank_E": int(re[i]),
                 "rank_T": int(rt[i]), "E": float(e[i]), "T": float(t[i])} for i in idx]

    return {"high_E_low_T": cell(he & ~ht, re), "low_E_high_T": cell(~he & ht, rt)}


def split_half_ceiling(loss_abl: np.ndarray, loss_ref: np.ndarray, k: int) -> float:
    """AUROC of T of the even clips for the top-k truth of T of the odd clips (and the reverse, averaged)."""
    d = loss_abl - loss_ref[None, :]
    a, b = d[:, 0::2].mean(axis=1), d[:, 1::2].mean(axis=1)
    return 0.5 * (M.auroc(a, truth_top_k(b, k)) + M.auroc(b, truth_top_k(a, k)))


def per_model(d: dict, size: np.ndarray | None, k: int) -> dict:
    e = np.asarray(d["B_C1"]["E_B"], dtype=np.float64)
    tt = d["T"]
    t = np.asarray(tt["T"], dtype=np.float64)
    dd = np.asarray(tt["D"], dtype=np.float64)
    truth = truth_top_k(t, k)
    ceiling = split_half_ceiling(np.asarray(tt["loss_abl"], dtype=np.float64),
                                 np.asarray(tt["loss_ref"], dtype=np.float64), k)
    top_e = truth_top_k(e, k)
    inter, union = int((top_e & truth).sum()), int((top_e | truth).sum())
    scores = c1_scores(d, size)
    ladder = {}
    for name in (m for m in LEADERBOARD if m in scores):
        s = np.asarray(scores[name], dtype=np.float64)
        a = M.auroc(s, truth)
        ladder[name] = {"AUROC": a, "signal_share": M.signal_share(a, ceiling), "AUPRC": M.auprc(s, truth),
                        "precision_at_k": M.precision_at_k(s, truth, k), "spearman": M.spearman(s, t),
                        "AUROC_vs_E": M.auroc(s, top_e)}
    return {"spearman_E_T": M.spearman(e, t), "overlap": inter, "jaccard": inter / union if union else float("nan"),
            "table": two_by_two(e, t, k), "n_T_positive": int((t > 0).sum()), "ceiling": ceiling,
            "spearman_D_T": M.spearman(dd, t), "spearman_D_E": M.spearman(dd, e), "ladder": ladder,
            "loss_ref_mean": float(np.mean(tt["loss_ref"]))}


def conclusion(res: dict) -> str:
    sp = res["spearman_E_T"]["pooled"]["mean"]
    ov = res["overlap"]["pooled"]["mean"]
    pt = res["pooled_mean_table"]
    best = max(res["methods"], key=lambda m: res["ladder"][m]["AUROC"]["pooled"]["mean"])
    return (f"Spearman between the ablation effect E and the task effect T is {sp:.2f} (mean over 10 models), "
            f"the top-{res['k']} sets of E and T share {ov:.1f} classes on average, and in the mean over models "
            f"{pt['high_E_low_T']} classes are high on E but low on T and {pt['low_E_high_T']} the reverse; "
            f"the best method against T is {plain(best)} with AUROC "
            f"{res['ladder'][best]['AUROC']['pooled']['mean']:.2f}.")


def analyze(data: dict) -> dict:
    """The `v1_1.z4` block of results.json (needs the task measurement of all 10 models)."""
    st_ = data["structure"]
    size = None if st_ is None else st_["n_krawedzi"].astype(np.float64)
    models = list(MODELS_POOLED)
    n = len(data["models"][models[0]]["B_C1"]["E_B"])
    k = threshold_count(n, TOP_FRACTION)
    per = {m: per_model(data["models"][m], size, k) for m in models}
    methods = list(per[models[0]]["ladder"])
    pooled_e = np.mean([data["models"][m]["B_C1"]["E_B"] for m in models], axis=0)
    pooled_t = np.mean([data["models"][m]["T"]["T"] for m in models], axis=0)
    d0 = data["models"][models[0]]["B_C1"]
    scalar = {key: _by_version({m: per[m][key] for m in models})
              for key in ("spearman_E_T", "overlap", "jaccard", "n_T_positive", "ceiling", "spearman_D_T",
                          "spearman_D_E", "loss_ref_mean")}
    ladder = {name: {metric: _by_version({m: per[m]["ladder"][name][metric] for m in models})
                     for metric in (*METRICS, "spearman", "AUROC_vs_E")} for name in methods}
    cells = ("high_E_high_T", "high_E_low_T", "low_E_high_T", "low_E_low_T")
    table = {c: _by_version({m: float(per[m]["table"][c]) for m in models}) for c in cells}
    res = {
        "k": k, "n_classes": n, "methods": methods, "models": models, "n_clips": int(
            len(data["models"][models[0]]["T"]["bodzce"])),
        **scalar, "table": table, "ladder": ladder,
        "pooled_mean_table": two_by_two(pooled_e, pooled_t, k),
        "pooled_mean_spearman_E_T": M.spearman(pooled_e, pooled_t),
        "off_diagonal": off_diagonal(pooled_e, pooled_t, k, d0["zrodlo"], d0["cel"]),
        "scatter": {"rank_E": _rank_desc(pooled_e).tolist(), "rank_T": _rank_desc(pooled_t).tolist()},
    }
    res["conclusion"] = conclusion(res)
    return res


# ---------------------------------------------------------------- report


def _f(x) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _cell(s: dict) -> str:
    lo, hi = s["ci_bootstrap"]
    return f"{_f(s['mean'])} [{_f(lo)}, {_f(hi)}]"


def _name(method: str) -> str:
    return f"{plain(method)} {DAGGER}"


def _cls(x: dict) -> str:
    return (f"`{x['source']}` to `{x['target']}` (rank on E {x['rank_E']}, rank on T {x['rank_T']}, "
            f"E {x['E']:.3g}, T {x['T']:.3g})")


def render_markdown(r: dict) -> str:
    z = r["v1_1"]["z4"]
    k = z["k"]
    L = [f"# Z4 {DAGGER}: a second truth at the level of the task", "",
         f"{DAGGER} Added after the study, exploratory, defined in `analysis-plan-v1.1.md` before it was run. "
         "It changes no preregistered decision. Mean over models with 95% bootstrap interval over models, seed "
         "20261101.", "",
         "`T(e)` = change of the flyvis training loss of the decoded optic flow on the Sintel clips of the study "
         f"({z['n_clips']} clips), ablated minus intact, mean over clips (`measure/task.py`). The task truth is the "
         f"top {k} classes by `T` (ties included, classes with `T <= 0` never).", "",
         f"**Conclusion {DAGGER}.** {z['conclusion']}", "",
         f"## E against T per model {DAGGER}", "",
         "| quantity | all 10 models | discovery (000-004) | replication (005-009) |", "|---|---|---|---|"]
    for key, label in (("spearman_E_T", "Spearman(E, T)"), ("overlap", f"top-{k} overlap of E and T (classes)"),
                       ("jaccard", "Jaccard of the two top sets"), ("n_T_positive", "classes with T > 0"),
                       ("ceiling", "AUROC ceiling of T (split-half over clips)"),
                       ("spearman_D_T", "Spearman(D, T), D = relative change of the decoded flow"),
                       ("spearman_D_E", "Spearman(D, E)")):
        s = z[key]
        L.append(f"| {label} {DAGGER} | {_cell(s['pooled'])} | {_cell(s['original'])} | {_cell(s['replication'])} |")
    L += ["", f"Spearman(E, T) of the mean over the 10 models: {_f(z['pooled_mean_spearman_E_T'])} {DAGGER}.", "",
          f"## High and low on E and on T {DAGGER}", "",
          "Counts of classes (high = top 61 by that quantity), mean over models, and for the mean over models.", "",
          "| cell | all 10 models | discovery | replication | mean over models (one table) |", "|---|---|---|---|---|"]
    for c, label in (("high_E_high_T", "high on E, high on T"), ("high_E_low_T", "high on E, low on T"),
                     ("low_E_high_T", "low on E, high on T"), ("low_E_low_T", "low on E, low on T")):
        t = z["table"][c]
        L.append(f"| {label} {DAGGER} | {_cell(t['pooled'])} | {_cell(t['original'])} | {_cell(t['replication'])} | "
                 f"{z['pooled_mean_table'][c]} |")
    od = z["off_diagonal"]
    L += ["", f"### High on E, low on T ({len(od['high_E_low_T'])} classes) {DAGGER}", ""]
    L += [f"- {_cls(x)}" for x in od["high_E_low_T"]] or ["(none)"]
    L += ["", f"### Low on E, high on T ({len(od['low_E_high_T'])} classes) {DAGGER}", ""]
    L += [f"- {_cls(x)}" for x in od["low_E_high_T"]] or ["(none)"]
    L += ["", f"## Method ladder against T {DAGGER}", "",
          f"AUROC and share of signal for the top-{k} truth by `T` (all 10 models); the last column is the AUROC "
          "of the same method against the truth by `E`.", "",
          "| method | AUROC vs T | share of signal | AUPRC | precision@61 | Spearman with T | AUROC vs E |",
          "|---|---|---|---|---|---|---|"]
    for name in z["methods"]:
        lad = z["ladder"][name]
        L.append(f"| {_name(name)} | {_cell(lad['AUROC']['pooled'])} | {_cell(lad['signal_share']['pooled'])} | "
                 f"{_cell(lad['AUPRC']['pooled'])} | {_cell(lad['precision_at_k']['pooled'])} | "
                 f"{_cell(lad['spearman']['pooled'])} | {_cell(lad['AUROC_vs_E']['pooled'])} |")
    L += ["", "![rank on E against rank on T](../figures/v1.1-z4-ranks-light.png)", "",
          f"Rank of each class on E against its rank on T (mean over models, 1 = largest) {DAGGER}; dashed lines "
          f"at rank {k}.", ""]
    return "\n".join(L).rstrip("\n") + "\n"


def ranks_figure(results: dict, plt, theme: dict):
    from ..figures import WIDTH_IN

    z = results["v1_1"]["z4"]
    re, rt = np.asarray(z["scatter"]["rank_E"]), np.asarray(z["scatter"]["rank_T"])
    k = z["k"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6))
    both = (re <= k) & (rt <= k)
    ax.scatter(re[~both], rt[~both], s=10, color=theme["muted"], lw=0, alpha=0.7, label="other classes")
    ax.scatter(re[both], rt[both], s=22, color=theme["S"], lw=0, label=f"top {k} on both")
    ax.axvline(k, color=theme["muted"], lw=0.8, ls=(0, (3, 3)))
    ax.axhline(k, color=theme["muted"], lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(len(re) + 10, -10)
    ax.set_ylim(len(rt) + 10, -10)
    ax.set_xlabel("rank on E (1 = largest ablation effect)")
    ax.set_ylabel("rank on T (1 = largest task effect)")
    ax.set_title(f"Ablation effect against task effect {DAGGER}, mean of 10 models", loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    return fig


def make_figures(results: dict, out: Path) -> list[Path]:
    from ..figures import THEMES, _mpl, _save

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme_name, theme in THEMES.items():
        plt = _mpl(theme)
        fig = ranks_figure(results, plt, theme)
        paths += _save(fig, out, "v1.1-z4-ranks", theme_name)
        plt.close(fig)
    return paths
