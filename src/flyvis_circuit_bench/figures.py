"""Figures from `results/results.json`: PNG and SVG, light and dark theme, colour-blind safe (Okabe-Ito).

Readability rule: every figure is 5 inches wide and every text is at least 15 pt, so after scaling the figure
to 400 px width (from 96 px per inch) text stays >= 12 pt. Output is deterministic (fixed jitter seed, no dates
in the files, fixed SVG id salt).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .names import ACCESS, ACCESS_OF
from .names import METHODS as NAME

WIDTH_IN = 5.0
MIN_FONT_PT = 15
SEED = 20261101
THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "S": "#D55E00", "P": "#0072B2", "A": "#E69F00", "other": "#999999", "ceiling": "#59636e",
              "access": ("#D55E00", "#E69F00", "#CC79A7", "#0072B2")},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "S": "#E69F00", "P": "#56B4E9", "A": "#F0E442", "other": "#8b949e", "ceiling": "#9198a1",
             "access": ("#D55E00", "#E69F00", "#CC79A7", "#56B4E9")},
}


def _mpl(theme: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": MIN_FONT_PT, "axes.titlesize": MIN_FONT_PT + 1,
        "axes.labelsize": MIN_FONT_PT, "xtick.labelsize": MIN_FONT_PT, "ytick.labelsize": MIN_FONT_PT,
        "legend.fontsize": MIN_FONT_PT, "axes.unicode_minus": False, "svg.hashsalt": "fcb",
        "svg.fonttype": "path", "figure.facecolor": theme["bg"], "axes.facecolor": theme["bg"],
        "savefig.facecolor": theme["bg"], "text.color": theme["fg"], "axes.labelcolor": theme["fg"],
        "axes.edgecolor": theme["muted"], "xtick.color": theme["fg"], "ytick.color": theme["fg"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": theme["grid"],
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    })
    return plt


def _save(fig, out: Path, stem: str, theme_name: str) -> list[Path]:
    paths = []
    for ext, meta in (("png", {"Software": None}), ("svg", {"Date": None, "Creator": None})):
        p = out / f"{stem}-{theme_name}.{ext}"
        fig.savefig(p, format=ext, dpi=200, bbox_inches="tight", pad_inches=0.12, metadata=meta)
        paths.append(p)
    return paths


# one marker shape per access group, so the groups differ without colour too
ACCESS_MARKERS = ("s", "D", "^", "o")
SHORT = {"P": "patching"}
LABEL_WIDTH = 0.14  # width of a label like "53%" in data units of the hero (15 pt on a 5 inch figure)


def share_rows(results: dict) -> list[dict]:
    """Leaderboard rows of the trained networks without the ceiling (100% by definition), by share of signal."""
    rows = [x for x in results["leaderboard"]["rows"] if x["condition"] == "C1" and x["method"] != "ceiling"]
    return sorted(rows, key=lambda x: x["signal_share"]["pooled"]["mean"])


def pct_label(x: float) -> str:
    return f"{100 * x:.0f}%"


def hero(results: dict, plt, theme: dict):
    """Share of the achievable signal of every method (mean over the trained models, bootstrap 95% CI),
    coloured by the access to the network the method needs."""
    rows = share_rows(results)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.6))
    for y, x in enumerate(rows):
        s = x["signal_share"]["pooled"]
        g = ACCESS.index(ACCESS_OF[x["method"]])
        color = theme["access"][g]
        ax.plot(s["ci_bootstrap"], [y, y], color=color, lw=5, solid_capstyle="butt", zorder=2)
        ax.scatter([s["mean"]], [y], color=color, marker=ACCESS_MARKERS[g], s=80, zorder=3,
                   edgecolor=theme["bg"], linewidth=0.8)
        lo, hi = s["ci_bootstrap"]
        # left of the interval; a row near 0% gets its label on the right, so it never crosses the chance line
        left = lo - LABEL_WIDTH - 0.03 > 0.02
        ax.text(lo - 0.03 if left else hi + 0.03, y, pct_label(s["mean"]), ha="right" if left else "left",
                va="center", fontsize=MIN_FONT_PT, color=theme["fg"])
    ax.axvline(0.0, color=theme["muted"], lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.axvline(1.0, color=theme["fg"], lw=1.4, zorder=1)
    ax.set_yticks(range(len(rows)), [SHORT.get(x["method"], NAME[x["method"]]) for x in rows])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%\nchance", "25%", "50%", "75%", "100%\nceiling"])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    n = len(rows[0]["signal_share"]["pooled"]["per_model"])
    ax.set_xlabel(f"share of the achievable signal, {n} models")
    ax.grid(axis="y", visible=False)
    keys = [plt.Line2D([], [], ls="none", marker=ACCESS_MARKERS[i], ms=9, color=theme["access"][i], label=name)
            for i, name in enumerate(ACCESS)]
    ax.legend(handles=keys, loc="lower left", bbox_to_anchor=(-0.62, 1.0), ncol=2, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0.2, title="the method needs", alignment="left")
    ax.set_title("The more access to the network,\nthe more you recover", loc="left", pad=96,
                 fontweight="bold", x=-0.62)
    return fig


def per_model(results: dict, plt, theme: dict):
    """AUROC of synapse count and attribution patching for every model, trained and untrained."""
    per = results["per_model_auroc"]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6))
    xs = {("C1", "S"): 0.0, ("C1", "P"): 1.35, ("C3", "S"): 2.95, ("C3", "P"): 4.3}
    for cond in ("C1", "C3"):
        models = sorted(m for m in per if cond in per[m])
        jitter = rng.uniform(-0.06, 0.06, size=len(models))
        for j, m in zip(jitter, models, strict=True):
            s, p = per[m][cond]["S"], per[m][cond]["P"]
            ax.plot([xs[(cond, "S")] + j, xs[(cond, "P")] + j], [s, p], color=theme["other"], lw=1.2,
                    alpha=0.7, zorder=1)
            ax.scatter([xs[(cond, "S")] + j], [s], s=46, color=theme["S"], zorder=3, edgecolor="none")
            ax.scatter([xs[(cond, "P")] + j], [p], s=46, color=theme["P"], zorder=3, edgecolor="none")
        ms = np.mean([per[m][cond]["S"] for m in models])
        mp = np.mean([per[m][cond]["P"] for m in models])
        ax.plot([xs[(cond, "S")] - 0.22, xs[(cond, "S")] + 0.22], [ms, ms], color=theme["fg"], lw=2.4, zorder=4)
        ax.plot([xs[(cond, "P")] - 0.22, xs[(cond, "P")] + 0.22], [mp, mp], color=theme["fg"], lw=2.4, zorder=4)
    t51 = next(q for q in results["replication"]["versions"]["pooled"]["quantities"] if q["key"] == "T5.1")
    lo, hi = t51["ci_bootstrap"]
    ax.annotate(f"gap +{t51['mean']:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]", xy=(1.2, 0.64), ha="center",
                va="center", fontsize=MIN_FONT_PT, color=theme["fg"])
    ax.set_xticks([0.675, 3.625], ["trained", "untrained"])
    ax.tick_params(axis="x", length=0, pad=8)
    keys = [plt.Line2D([], [], ls="none", marker="o", ms=8, color=theme[k], label=name)
            for k, name in (("S", "synapse count"), ("P", "attribution patching"))]
    ax.legend(handles=keys, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.1,
              columnspacing=1.0, borderaxespad=0.2)
    # below every point of the untrained group (its lowest AUROC is far above 0.75), so it never covers data
    ax.text(3.625, 0.70, "patching at\nthe ceiling", ha="center", va="center", color=theme["muted"],
            fontsize=MIN_FONT_PT)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlim(-0.55, 4.85)
    ax.set_ylabel("AUROC vs ablation truth")
    ax.grid(axis="x", visible=False)
    n = len([m for m in per if "C1" in per[m]])
    ax.set_title(f"Synapse count vs attribution\npatching, {n} models", pad=34, loc="left", fontweight="bold")
    return fig


FOREST = (("T5.1", "patching\nvs synapse count"), ("D^M1", "gradient\nvs synapse count"),
          ("D^M3", "activity x weight\nvs synapse count"), ("T5.3", "synapse count,\ntrained vs untrained"),
          ("T5.4", "patching vs\nsynapse count,\nsecond stimulus set"))


def forest(results: dict, plt, theme: dict):
    """The preregistered replication decisions: all 10 models, discovery 000-004, replication 005-009."""
    vers = results["replication"]["versions"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.4))
    styles = (("original", "discovery", theme["fg"], 0.24), ("pooled", "all 10", theme["P"], 0.0),
              ("replication", "replication", theme["S"], -0.24))
    for i, (key, _label) in enumerate(FOREST):
        y = len(FOREST) - 1 - i
        for ver, name, color, dy in styles:
            q = next(x for x in vers[ver]["quantities"] if x["key"] == key)
            ax.plot(q["ci_t"], [y + dy] * 2, color=color, lw=1.4, alpha=0.6)
            ax.plot(q["ci_bootstrap"], [y + dy] * 2, color=color, lw=4.0, solid_capstyle="butt")
            ax.scatter([q["mean"]], [y + dy], color=color, s=40, zorder=3,
                       label=name if i == 0 else None, edgecolor=theme["bg"], linewidth=0.8)
    ax.axvline(0, color=theme["muted"], lw=1.2)
    ax.set_yticks(range(len(FOREST)), [lab for _, lab in reversed(FOREST)])
    ax.set_xlabel("difference in AUROC")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    ax.set_title("Replication: every preregistered\ndecision holds", loc="left", pad=44, fontweight="bold")
    ax.text(1.0, -0.2, "thick: bootstrap CI, thin: t CI", transform=ax.transAxes, ha="right", va="top",
            color=theme["muted"], fontsize=MIN_FONT_PT)
    return fig


FIGURES = {"hero": hero, "per-model": per_model, "replication": forest}


def make_all(results: dict, out: Path) -> list[Path]:
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
