"""v1.1 analysis Z7 (added after the study, exploratory): patching with a second-order correction.

Definition: `docs/analysis-plan-v1.1.md`, section Z7 and its clarification of 2026-11-13; measurement:
`measure/hvp.py` (`fcb measure --what hvp`). Nothing here changes a preregistered decision.

* Score `P+HVP` per class: mean over readout directions and stimuli of `(first + 0.5 second)^2 / ||r||^2`, with
  `first` the patching change and `second` the diagonal entry of the Hessian along the class (same directions, same
  aggregation as patching `P`).
* The measurement uses 8 of the 40 stimuli of B (indices fixed by the plan, second clarification of 2026-11-13).
  Rows: `P` = patching on the same 8 stimuli, the first-order term of the same run (the fair comparison with
  `P+HVP`), `P+HVP`, and `P (40)` = patching of the package on all 40 stimuli, for context only.
* Metrics as in the study at the main threshold (top 10%, k = 61 of 604): AUROC against the truth `E` (40 stimuli),
  share of signal with the split-half ceiling, AUPRC, precision@k, Spearman with `E`.
* Gap closed: (share(P+HVP) - share(P)) / (1 - share(P)), from the means over models, `P` on the same 8 stimuli.
* Check: Spearman between patching of the package (40 stimuli) and the first-order term of this run (8 stimuli).

Statistics as in the study: mean over models, bootstrap and t interval over models, seed 20261101.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..data import MODELS_POOLED
from .z1 import DAGGER, _by_version, threshold_count, truth_top_k

KEY = "z7"
TOP_FRACTION = 0.10
METRICS = ("AUROC", "signal_share", "AUPRC", "precision_at_k", "spearman")
ROWS = ("P", "P+HVP", "P40")
COMPARED = ("P", "P+HVP")


def available(data: dict) -> bool:
    """True when the hvp measurement exists for every one of the 10 models."""
    return all("HVP" in data["models"].get(m, {}) for m in MODELS_POOLED)


def per_model(d: dict, k: int) -> dict:
    b = d["B_C1"]
    e = np.asarray(b["E_B"], dtype=np.float64)
    truth = truth_top_k(e, k)
    ceiling = M.auroc(np.asarray(b["E_B1"], dtype=np.float64), truth_top_k(b["E_B2"], k))
    h = d["HVP"]
    scores = {"P": np.asarray(h["P_chk"], dtype=np.float64), "P+HVP": np.asarray(h["PH"], dtype=np.float64),
              "P40": np.asarray(b["P"], dtype=np.float64)}
    out: dict = {"ceiling": ceiling, "methods": {}}
    for name, s in scores.items():
        a = M.auroc(s, truth)
        out["methods"][name] = {"AUROC": a, "signal_share": M.signal_share(a, ceiling), "AUPRC": M.auprc(s, truth),
                                "precision_at_k": M.precision_at_k(s, truth, k), "spearman": M.spearman(s, e)}
    p, ph = out["methods"]["P"], out["methods"]["P+HVP"]
    out["diff"] = {"AUROC": ph["AUROC"] - p["AUROC"], "signal_share": ph["signal_share"] - p["signal_share"]}
    out["check_P"] = M.spearman(scores["P40"], scores["P"])
    # size of the correction: fraction of the corrected term that is the second-order part, and its sign
    cross = np.asarray(h["cross"], dtype=np.float64)
    out["cross_fraction"] = float(np.nansum(cross) / np.nansum(scores["P+HVP"]))
    return out


def gap_closed(share_p: float, share_ph: float) -> float:
    return (share_ph - share_p) / (1.0 - share_p) if share_p < 1.0 else float("nan")


def conclusion(res: dict) -> str:
    p = res["methods"]["P"]["signal_share"]["pooled"]["mean"]
    ph = res["methods"]["P+HVP"]["signal_share"]["pooled"]["mean"]
    p40 = res["methods"]["P40"]["signal_share"]["pooled"]["mean"]
    d = res["diff"]["AUROC"]["pooled"]
    lo, hi = d["ci_bootstrap"]
    verdict = ("the interval of the AUROC change excludes zero" if lo > 0 or hi < 0
               else "the interval of the AUROC change includes zero")
    return (f"On the same {res['n_stimuli']} stimuli patching reaches {p:.3f} of the achievable signal and patching "
            f"with the Hessian correction {ph:.3f} (mean over 10 models; patching on all 40 stimuli {p40:.3f}); "
            f"the AUROC changes by {d['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"({verdict}), which closes {res['gap_closed']['pooled'] * 100:.0f}% of the gap to the ceiling.")


def analyze(data: dict) -> dict:
    """The `v1_1.z7` block of results.json (needs the hvp measurement of all 10 models)."""
    models = list(MODELS_POOLED)
    n = len(data["models"][models[0]]["B_C1"]["E_B"])
    k = threshold_count(n, TOP_FRACTION)
    per = {m: per_model(data["models"][m], k) for m in models}
    methods = {name: {metric: _by_version({m: per[m]["methods"][name][metric] for m in models})
                      for metric in METRICS} for name in ROWS}
    diff = {key: _by_version({m: per[m]["diff"][key] for m in models}) for key in ("AUROC", "signal_share")}
    gap = {}
    for version, label in (("pooled", "pooled"), ("original", "original"), ("replication", "replication")):
        sp, sh = methods["P"]["signal_share"][version]["mean"], methods["P+HVP"]["signal_share"][version]["mean"]
        gap[label] = gap_closed(sp, sh)
    n_stim = {int(data["models"][m]["HVP"].get("meta", {}).get("n_stimuli", 0)) for m in models}
    res = {"k": k, "n_classes": n, "n_stimuli": n_stim.pop() if len(n_stim) == 1 else None,
           "models": models, "methods": methods, "diff": diff, "gap_closed": gap,
           "ceiling": _by_version({m: per[m]["ceiling"] for m in models}),
           "check_P_spearman": _by_version({m: per[m]["check_P"] for m in models}),
           "cross_fraction": _by_version({m: per[m]["cross_fraction"] for m in models}),
           "meta": {m: _meta(data["models"][m]["HVP"]) for m in models}}
    res["conclusion"] = conclusion(res)
    return res


def _meta(h: dict) -> dict:
    meta = h.get("meta", {})
    return {key: meta.get(key) for key in ("backend", "K", "n_stimuli", "stimuli_index")}


# ---------------------------------------------------------------- report


def _f(x) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _cell(s: dict) -> str:
    lo, hi = s["ci_bootstrap"]
    return f"{_f(s['mean'])} [{_f(lo)}, {_f(hi)}]"


def _cell_t(s: dict) -> str:
    lo, hi = s["ci_t"]
    return f"[{_f(lo)}, {_f(hi)}]"


def _name(row: str) -> str:
    return {"P": "Patching, same 8 stimuli", "P40": "Patching, 40 stimuli of the package (context)"}.get(
        row, f"Patching + Hessian correction {DAGGER}")


def render_markdown(r: dict) -> str:
    z = r["v1_1"]["z7"]
    k = z["k"]
    L = [f"# Z7 {DAGGER}: a second-order correction for attribution patching", "",
         f"{DAGGER} Added after the study, exploratory, defined in `analysis-plan-v1.1.md` (Z7 and its "
         "clarification of 2026-11-13) before it was run. It changes no preregistered decision. Mean over models "
         "with 95% bootstrap interval over models, seed 20261101.", "",
         "Patching is the first-order change of a readout when one connection class is scaled to zero. The correction "
         "adds half the second derivative along the same class (the diagonal of the Hessian, one Hessian-vector "
         "product per class and readout direction), so the score is the second-order Taylor expansion of the "
         "ablation. Same readout directions and aggregation as patching.", "",
         f"**Conclusion {DAGGER}.** {z['conclusion']}", "",
         f"## Leaderboard rows (top {k} truth, 10 models) {DAGGER}", "",
         "| method | AUROC | share of signal | AUPRC | precision@61 | Spearman with E |", "|---|---|---|---|---|---|"]
    for row in ROWS:
        m = z["methods"][row]
        L.append(f"| {_name(row)} | " + " | ".join(_cell(m[x]["pooled"]) for x in METRICS[:4]) + f" | "
                 f"{_cell(m['spearman']['pooled'])} |")
    L += ["", f"## Change by the correction {DAGGER}", "",
          "| quantity | all 10 models | t interval | discovery (000-004) | replication (005-009) |",
          "|---|---|---|---|---|"]
    for key, label in (("AUROC", "AUROC change by the correction"), ("signal_share", "share of signal change")):
        s = z["diff"][key]
        L.append(f"| {label} {DAGGER} | {_cell(s['pooled'])} | {_cell_t(s['pooled'])} | {_cell(s['original'])} | "
                 f"{_cell(s['replication'])} |")
    L += ["", "| share of signal | all 10 models | t interval | discovery | replication |", "|---|---|---|---|---|"]
    for row in ROWS:
        s = z["methods"][row]["signal_share"]
        L.append(f"| {_name(row)} | {_cell(s['pooled'])} | {_cell_t(s['pooled'])} | {_cell(s['original'])} | "
                 f"{_cell(s['replication'])} |")
    g = z["gap_closed"]
    L += ["", f"Gap to the ceiling closed {DAGGER}: {g['pooled']:.1%} (discovery {g['original']:.1%}, replication "
          f"{g['replication']:.1%}).", "",
          f"Agreement of the two patching rows {DAGGER}: Spearman between patching of the package (40 stimuli) and "
          f"the first-order term of this run ({z['n_stimuli']} stimuli): {_cell(z['check_P_spearman']['pooled'])}.", ""]
    return "\n".join(L).rstrip("\n") + "\n"


def make_figures(results: dict, out: Path) -> list[Path]:
    """No figure (optional in the plan)."""
    return []
