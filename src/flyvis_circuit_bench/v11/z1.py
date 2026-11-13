"""v1.1 analysis Z1 (added after the study, exploratory): sensitivity to the threshold, continuous metrics.

Definitions are those of `docs/analysis-plan-v1.1.md`, section Z1. Nothing here changes a preregistered decision.

* Thresholds: top 5%, 10%, 15%, 20% of the classes, k = ceil(fraction * n) (31, 61, 91, 121 for n = 604).
* Per method and threshold: AUROC, share of signal with the ceiling recomputed at that threshold, AUPRC,
  precision@k with k = the threshold count. The truth is the top-k classes by E (ties at the cut included,
  zero-effect classes never), as in `metrics.binary_truth`.
* Spearman of the score with E over all classes.
* Curve: precision@k for k = 10, 20, ..., 200, where the truth of each k is the top-k classes (so the curve passes
  through the table values at k = 31, 61, ... only for the four thresholds, the rest are the same rule at other k).
* Swap: the sign of the difference of mean AUROC of two methods differs between two thresholds and both 95%
  bootstrap intervals of the difference exclude zero; pairs whose interval includes zero at either threshold are
  "not separated".

Statistics as in the study: mean over models, bootstrap and t interval over models, seed 20261101.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..analysis import LEADERBOARD, VERSIONS, c1_scores, ceilings
from ..names import METHODS as NAME
from ..stats import bootstrap_ci, summarize

KEY = "z1"
FRACTIONS = (0.05, 0.10, 0.15, 0.20)
CURVE_K = tuple(range(10, 201, 10))
METRICS = ("AUROC", "signal_share", "AUPRC", "precision_at_k")
DAGGER = "†"


def plain(method: str) -> str:
    return NAME[method].removesuffix(DAGGER).rstrip()


def threshold_count(n_classes: int, fraction: float) -> int:
    return math.ceil(round(fraction * n_classes, 9))


def truth_top_k(e: np.ndarray, k: int) -> np.ndarray:
    """Top-k classes by E; ties at the cut included, zero-effect classes never (same rule as binary_truth)."""
    e = np.asarray(e, dtype=float)
    cut = np.sort(e)[::-1][k - 1]
    return (e > 0) if cut <= 0 else (e >= cut)


def per_model_values(d: dict, size: np.ndarray | None, ks: tuple[int, ...], curve_ks: tuple[int, ...]) -> dict:
    """Everything Z1 needs from one trained model: per method and threshold the four metrics, the Spearman with E
    and the precision@k curve; plus the ceiling AUROC per threshold."""
    b = d["B_C1"]
    e = np.asarray(b["E_B"], dtype=np.float64)
    scores = c1_scores(d, size)
    n = len(e)
    out: dict = {"ceiling": {}, "method": {}}
    for k in ks:
        out["ceiling"][k] = M.auroc(np.asarray(b["E_B1"], dtype=np.float64), truth_top_k(b["E_B2"], k))
    for name in LEADERBOARD:
        s = np.asarray(scores[name], dtype=np.float64)
        row: dict = {"spearman": M.spearman(s, e), "k": {}, "curve": {}}
        for k in ks:
            y = truth_top_k(e, k)
            a = M.auroc(s, y)
            row["k"][k] = {"AUROC": a, "signal_share": M.signal_share(a, out["ceiling"][k]),
                           "AUPRC": M.auprc(s, y), "precision_at_k": M.precision_at_k(s, y, k)}
        for k in curve_ks:
            row["curve"][k] = M.precision_at_k(s, truth_top_k(e, k), k) if k <= n else float("nan")
        out["method"][name] = row
    return out


def _compact(s) -> dict:
    return {"mean": s.mean, "ci_bootstrap": list(s.boot), "n": s.n}


def _by_version(per_model: dict[str, float], compact: bool = False) -> dict:
    out = {}
    for v, plan in VERSIONS.items():
        s = summarize(per_model, plan)
        out[v] = _compact(s) if compact else s.to_dict()
    return out


def _paired_ci(per_a: dict[str, float], per_b: dict[str, float], plan) -> tuple[float, tuple[float, float]]:
    models = [m for m in sorted(plan) if m in per_a and m in per_b]
    diff = [per_a[m] - per_b[m] for m in models]
    return float(np.mean(diff)), bootstrap_ci(diff)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def pair_changes(auroc: dict[str, dict[int, dict[str, float]]], ks: tuple[int, ...], plan) -> dict:
    """Swapped and not-separated method pairs between thresholds, for one set of models.

    `auroc[method][k][model]`. Only pairs whose sign of the mean AUROC difference differs between two thresholds
    are candidates; they are swaps when both bootstrap intervals exclude zero, else not separated.
    """
    swaps, not_separated = [], []
    for a, b in itertools.combinations(LEADERBOARD, 2):
        stat = {k: _paired_ci(auroc[a][k], auroc[b][k], plan) for k in ks}
        for k1, k2 in itertools.combinations(ks, 2):
            (m1, ci1), (m2, ci2) = stat[k1], stat[k2]
            if _sign(m1) == _sign(m2):
                continue
            sep = all(ci[0] > 0 or ci[1] < 0 for ci in (ci1, ci2))
            item = {"a": a, "b": b, "k": [k1, k2], "diff": [m1, m2], "ci": [list(ci1), list(ci2)]}
            (swaps if sep else not_separated).append(item)
    return {"swaps": swaps, "not_separated": not_separated}


def conclusion(res: dict) -> str:
    """One sentence, generated from the numbers (so it cannot drift from them)."""
    p = res["pairs"]["pooled"]
    n_pairs = math.comb(len(res["methods"]), 2)
    sp = {m: res["spearman"][m]["pooled"]["mean"] for m in res["methods"]}
    share = {m: [r["signal_share"]["pooled"]["mean"] for r in res["table"] if r["method"] == m] for m in res["methods"]}
    lo, hi = min(min(v) for v in share.values()), max(max(v) for v in share.values())
    best = max(sp, key=sp.get)
    n_sw, n_ns = len(p["swaps"]), len(p["not_separated"])
    return (f"Across the four thresholds {n_sw} swap(s) and {n_ns} not-separated sign change(s) occur among "
            f"{n_pairs} method pairs (all 10 models), shares of signal stay within {lo:.2f} to {hi:.2f}, and the "
            f"highest Spearman with E is {sp[best]:.2f} ({plain(best)}).")


def analyze(data: dict) -> dict:
    """The `v1_1.z1` block of results.json."""
    st_ = data["structure"]
    size = None if st_ is None else st_["n_krawedzi"].astype(np.float64)
    models = sorted(ceilings(data))
    n = len(data["models"][models[0]]["B_C1"]["E_B"])
    ks = tuple(threshold_count(n, f) for f in FRACTIONS)
    curve_ks = tuple(k for k in CURVE_K if k <= n)
    per = {m: per_model_values(data["models"][m], size, ks, curve_ks) for m in models}

    table = []
    for name in LEADERBOARD:
        for f, k in zip(FRACTIONS, ks, strict=True):
            row = {"method": name, "fraction": f, "k": k}
            for metric in METRICS:
                row[metric] = _by_version({m: per[m]["method"][name]["k"][k][metric] for m in models})
            table.append(row)
    ceiling = [{"fraction": f, "k": k, **{"AUROC": _by_version({m: per[m]["ceiling"][k] for m in models})}}
               for f, k in zip(FRACTIONS, ks, strict=True)]
    spearman = {name: _by_version({m: per[m]["method"][name]["spearman"] for m in models})
                for name in LEADERBOARD}
    curve = {name: {str(k): _by_version({m: per[m]["method"][name]["curve"][k] for m in models}, compact=True)
                    for k in curve_ks} for name in LEADERBOARD}
    auroc = {name: {k: {m: per[m]["method"][name]["k"][k]["AUROC"] for m in models} for k in ks}
             for name in LEADERBOARD}
    pairs = {v: pair_changes(auroc, ks, plan) for v, plan in VERSIONS.items()}
    res = {"dagger": DAGGER, "thresholds": [{"fraction": f, "k": k} for f, k in zip(FRACTIONS, ks, strict=True)],
           "n_classes": n, "methods": list(LEADERBOARD), "table": table, "ceiling": ceiling, "spearman": spearman,
           "curve_k": list(curve_ks), "curve": curve, "pairs": pairs}
    res["conclusion"] = conclusion(res)
    return res


# ---------------------------------------------------------------- report


def _f(x) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _name(method: str) -> str:
    """Method name with exactly one dagger (two rows already carry it from the v1.0.1 table)."""
    return f"{plain(method)} {DAGGER}"


def _cell(s: dict) -> str:
    lo, hi = s["ci_bootstrap"]
    return f"{_f(s['mean'])} [{_f(lo)}, {_f(hi)}]"


def _pair_line(x: dict) -> str:
    (k1, k2), (d1, d2) = x["k"], x["diff"]
    return (f"- {plain(x['a'])} vs {plain(x['b'])}: AUROC difference {_f(d1)} at top {k1} "
            f"[{_f(x['ci'][0][0])}, {_f(x['ci'][0][1])}], {_f(d2)} at top {k2} "
            f"[{_f(x['ci'][1][0])}, {_f(x['ci'][1][1])}] {DAGGER}")


def render_markdown(r: dict) -> str:
    z = r["v1_1"]["z1"]
    ks = [t["k"] for t in z["thresholds"]]
    L = [f"# Z1 {DAGGER}: sensitivity to the threshold, continuous metrics", "",
         f"{DAGGER} Added after the study, exploratory, defined in `analysis-plan-v1.1.md` before it was run. "
         "It changes no preregistered decision. Mean over models with 95% bootstrap interval over models, "
         "seed 20261101; truth = the top-k classes by E (ties at the cut included).", "",
         f"**Conclusion {DAGGER}.** {z['conclusion']}", ""]
    for v, title in (("pooled", "all 10 models"), ("original", "discovery (000-004)"),
                     ("replication", "replication (005-009)")):
        L += [f"## Thresholds, {title} {DAGGER}", ""]
        head = ["| method | " + " | ".join(f"top {k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for metric, label in (("AUROC", "AUROC"), ("signal_share", "share of signal (ceiling per threshold)"),
                              ("AUPRC", "AUPRC"), ("precision_at_k", "precision@k, k = threshold")):
            L += [f"### {label} {DAGGER}", "", *head]
            if metric == "AUROC":
                L.append(f"| truth ceiling {DAGGER} | " + " | ".join(_cell(c["AUROC"][v]) for c in z["ceiling"]) + " |")
            for name in z["methods"]:
                cells = [_cell(row[metric][v]) for row in z["table"] if row["method"] == name]
                L.append(f"| {_name(name)} | " + " | ".join(cells) + " |")
            L.append("")
        if v == "pooled":
            L += [f"### Spearman with E, all {z['n_classes']} classes {DAGGER}", "",
                  "| method | all 10 models | discovery | replication |", "|---|---|---|---|"]
            for name in z["methods"]:
                s = z["spearman"][name]
                L.append(f"| {_name(name)} | {_cell(s['pooled'])} | {_cell(s['original'])} | "
                         f"{_cell(s['replication'])} |")
            L.append("")
    L += [f"## Ordering changes {DAGGER}", "",
          "A pair swaps when the sign of the mean AUROC difference differs between two thresholds and both 95% "
          "bootstrap intervals of the difference exclude zero; a sign change with an interval that includes zero "
          "at either threshold is listed as not separated.", ""]
    for v, title in (("pooled", "all 10 models"), ("original", "discovery (000-004)"),
                     ("replication", "replication (005-009)")):
        p = z["pairs"][v]
        L += [f"### {title} {DAGGER}", "", f"Swapped pairs: {len(p['swaps'])}.", ""]
        L += [_pair_line(x) for x in p["swaps"]] + ([""] if p["swaps"] else [])
        L += [f"Not separated: {len(p['not_separated'])}.", ""]
        L += [_pair_line(x) for x in p["not_separated"]] + ([""] if p["not_separated"] else [])
    L += [f"## Precision@k curve {DAGGER}", "",
          "![precision at k](../figures/v1.1-z1-curve-light.png)", "",
          f"Mean precision@k over all 10 models for k = {z['curve_k'][0]}, ..., {z['curve_k'][-1]}; "
          "the truth of each k is the top-k classes by E. Values per method and k, with intervals, are under "
          "`v1_1.z1.curve` in `results/results.json`.", ""]
    return "\n".join(L).rstrip("\n") + "\n"


# ---------------------------------------------------------------- figure


def curve_figure(results: dict, plt, theme: dict):
    """precision@k of every method against k (mean over models, bootstrap band), top 4 thresholds marked."""
    from ..figures import ACCESS_MARKERS, WIDTH_IN
    from ..names import ACCESS, ACCESS_OF

    z = results["v1_1"]["z1"]
    ks = z["curve_k"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.0))
    styles = ("-", "--", ":")  # methods of one access group differ by line style (and the group by colour, marker)
    used: dict[int, int] = {}
    handles = []
    for name in z["methods"]:
        g = ACCESS.index(ACCESS_OF[name])
        color = theme["access"][g]
        ls = styles[used.get(g, 0) % len(styles)]
        used[g] = used.get(g, 0) + 1
        c = z["curve"][name]
        mean = [c[str(k)]["pooled"]["mean"] for k in ks]
        lo = [c[str(k)]["pooled"]["ci_bootstrap"][0] for k in ks]
        hi = [c[str(k)]["pooled"]["ci_bootstrap"][1] for k in ks]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(ks, mean, color=color, lw=1.8, ls=ls, marker=ACCESS_MARKERS[g], ms=5)
        handles.append(plt.Line2D([], [], color=color, ls=ls, lw=1.8, marker=ACCESS_MARKERS[g], ms=7,
                                  label=plain(name)))
    for t in z["thresholds"]:
        ax.axvline(t["k"], color=theme["muted"], lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(ks[0] - 4, ks[-1] + 4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("k (size of the truth and of the cut)")
    ax.set_ylabel("precision@k")
    ax.set_title(f"Precision@k {DAGGER}, 10 models", loc="left", fontweight="bold")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, handletextpad=0.4,
              handlelength=3.0)
    return fig


def make_figures(results: dict, out: Path) -> list[Path]:
    from ..figures import THEMES, _mpl, _save

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme_name, theme in THEMES.items():
        plt = _mpl(theme)
        fig = curve_figure(results, plt, theme)
        paths += _save(fig, out, "v1.1-z1-curve", theme_name)
        plt.close(fig)
    return paths
