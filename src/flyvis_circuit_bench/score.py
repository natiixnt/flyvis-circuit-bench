"""Score your own method against the ablation truth and place it on the leaderboard.

A method gives every model a score per connection class (604 numbers, larger = more important), computed
without the truth. `method_inputs` returns what the package offers a method (connectome and weights, no truth);
a method may also use anything else it computes itself, e.g. from the flyvis models.

    from flyvis_circuit_bench import score
    inputs = score.method_inputs("data/measurements-v1.1.0")
    scores = {m: x["sum_abs_weight"] / x["n_edges"] for m, x in inputs.items()}
    result = score.evaluate(scores, "data/measurements-v1.1.0")
    print(score.leaderboard_row("mean |weight| per edge", result))

From the shell: save the scores with `numpy.savez(path, **scores)` (one array per model id) and run
`fcb score --data data/measurements-v1.1.0 --scores path.npz --name "my method"`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from . import analysis
from . import metrics as M
from .data import MODELS_POOLED, load


def _data(data) -> dict:
    return data if isinstance(data, dict) else load(Path(data))


def method_inputs(data) -> dict[str, dict[str, np.ndarray]]:
    """Per trained model: source and target cell type, synapse count, sum of |weights| and number of edges of
    each class. Nothing derived from the ablations is included."""
    d = _data(data)
    n_edges = d["structure"]["n_krawedzi"].astype(np.float64)
    out = {}
    for m in MODELS_POOLED:
        b = d["models"][m]["B_C1"]
        out[m] = {"source": b["zrodlo"].astype(str), "target": b["cel"].astype(str),
                  "synapse_count": np.asarray(b["S"], np.float64),
                  "sum_abs_weight": np.asarray(b["A"], np.float64), "n_edges": n_edges.copy()}
    return out


def check_scores(scores, n_classes: int) -> dict[str, np.ndarray]:
    """{model id: 1-d finite float array of length n_classes} from a mapping or from one shared array."""
    if not isinstance(scores, Mapping):
        scores = dict.fromkeys(MODELS_POOLED, scores)
    missing = [m for m in MODELS_POOLED if m not in scores]
    if missing:
        raise ValueError(f"scores missing for models {', '.join(missing)} (need {', '.join(MODELS_POOLED)})")
    extra = sorted(set(scores) - set(MODELS_POOLED))
    if extra:
        raise ValueError(f"unknown model ids in scores: {', '.join(map(str, extra))}")
    out = {}
    for m in MODELS_POOLED:
        try:
            x = np.asarray(scores[m], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"scores of model {m} are not numbers: {e}") from None
        if x.shape != (n_classes,):
            raise ValueError(f"scores of model {m} have shape {x.shape}, expected ({n_classes},)")
        if not np.isfinite(x).all():
            raise ValueError(f"scores of model {m} contain nan or inf")
        out[m] = x
    return out


def evaluate(scores, data) -> dict:
    """Leaderboard metrics of a method: AUROC, share of the achievable signal, AUPRC and precision@61, each as
    mean over the 10 trained models with both 95% CIs, and separately for models 000-004 and 005-009."""
    d = _data(data)
    ceil = analysis.ceilings(d, "C1")
    e = {m: d["models"][m]["B_C1"]["E_B"] for m in MODELS_POOLED}
    s = check_scores(scores, len(e[MODELS_POOLED[0]]))
    per = {m: M.method_metrics(s[m], e[m], ceil[m]) for m in MODELS_POOLED}
    return {"per_model": per, **analysis.summaries(per)}


def load_scores(path: Path):
    """Scores from `.npz` (one array per model id, or a single array for every model), `.npy` (one array for
    every model, or 10 rows in model order) or `.json` ({model id: list} or one list)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return next(iter(arrays.values())) if len(arrays) == 1 else arrays
    if path.suffix == ".npy":
        x = np.load(path, allow_pickle=False)
        return dict(zip(MODELS_POOLED, x, strict=True)) if x.ndim == 2 else x
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported scores file {path.name}: use .npz, .npy or .json")


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def leaderboard_row(name: str, result: dict, cost: str = "?") -> str:
    """One row in the format of the README leaderboard (paste it into your pull request)."""
    a = result["AUROC"]["pooled"]
    lo, hi = a["ci_bootstrap"]
    return (f"| {name.replace('|', chr(92) + '|')} | {cost} | {a['mean']:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"{_pct(result['signal_share']['pooled']['mean'])} | {result['AUROC']['original']['mean']:.3f} | "
            f"{result['AUROC']['replication']['mean']:.3f} | {result['AUPRC']['pooled']['mean']:.3f} | "
            f"{result['precision_at_61']['pooled']['mean']:.3f} |")
