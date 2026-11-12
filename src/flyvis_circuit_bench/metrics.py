"""Ranking metrics against the ablation ground truth (pure numpy and scipy).

Ground truth for one model and condition: E(e), the mean over stimuli of ||r_e(b) - r(b)||^2 / ||r(b)||^2 after
zeroing the synaptic scale of connection class e (604 type-to-type classes). Binary truth: the top 10% of
classes by E (ceil(0.1 * 604) = 61 classes, ties at the threshold included; when the 61st value is <= 0 only
classes with E > 0 count). A method is a score per class computed without the truth.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

TOP_FRACTION = 0.10


def binary_truth(e: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    """Top `fraction` of classes by E; ties at the threshold are included, zero-effect classes never are."""
    e = np.asarray(e, dtype=float)
    k = math.ceil(fraction * len(e))
    threshold = np.sort(e)[::-1][k - 1]
    return (e > 0) if threshold <= 0 else (e >= threshold)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC as the Mann-Whitney statistic (tied scores count one half); nan without both classes."""
    y = np.asarray(label, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(np.asarray(score, dtype=float), method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision: sum over thresholds of (R_i - R_{i-1}) * P_i; equal scores form one threshold."""
    y = np.asarray(label, dtype=bool)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    ap, tp, fp, r_prev = 0.0, 0, 0, 0.0
    for threshold in np.unique(s)[::-1]:
        m = s == threshold
        tp += int(y[m].sum())
        fp += int((~y[m]).sum())
        r = tp / y.sum()
        ap += (r - r_prev) * tp / (tp + fp)
        r_prev = r
    return float(ap)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(st.spearmanr(a, b).statistic)


def spearman_positives(score: np.ndarray, e: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """Measure c1: Spearman of the score with E inside the top-10% classes only.

    It is the one measure that does not saturate in the untrained condition C3 (AUROC of P is ~1.000 there).
    nan when fewer than 3 positives or either vector is constant on them.
    """
    y = binary_truth(e, fraction)
    w, ee = np.asarray(score, dtype=float)[y], np.asarray(e, dtype=float)[y]
    if y.sum() < 3 or np.ptp(w) == 0 or np.ptp(ee) == 0:
        return float("nan")
    return spearman(w, ee)


MEASURES = ("AUROC", "AUPRC", "Spearman", "c1")


def measure(name: str, score: np.ndarray, e: np.ndarray) -> float:
    """One measure of a score against the continuous truth E."""
    score = np.asarray(score, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    if name == "AUROC":
        return auroc(score, binary_truth(e))
    if name == "AUPRC":
        return auprc(score, binary_truth(e))
    if name == "Spearman":
        return spearman(score, e)
    if name == "c1":
        return spearman_positives(score, e)
    raise ValueError(f"unknown measure {name!r}")


def ceiling_fraction(auroc_method: float, auroc_ceiling: float) -> float:
    """(AUROC - 0.5) / (ceiling - 0.5): share of the above-chance headroom a method reaches."""
    if not auroc_ceiling > 0.5:
        return float("nan")
    return (auroc_method - 0.5) / (auroc_ceiling - 0.5)


def signal_share(auroc_method: float, auroc_ceiling: float) -> float:
    """Share of the achievable signal: (AUROC - 0.5) / (ceiling - 0.5); 0 = chance, 1 = as good as the truth."""
    return ceiling_fraction(auroc_method, auroc_ceiling)


def precision_at_k(score: np.ndarray, label: np.ndarray, k: int | None = None) -> float:
    """Share of the k top-scored classes that are truth positives; k defaults to the number of positives.

    With the 604 classes and the top-10% truth this is "how many of the 61 most important connections are in
    the method's top 61". Scores tied at the cut share the remaining slots in proportion (the expected value
    under random tie-breaking), so the value does not depend on the order of the classes.
    """
    y = np.asarray(label, dtype=bool)
    s = np.asarray(score, dtype=float)
    k = int(y.sum()) if k is None else int(k)
    if k <= 0 or k > len(s):
        return float("nan")
    cut = np.sort(s)[::-1][k - 1]
    above, tied = s > cut, s == cut
    slots = k - int(above.sum())
    hits = int(y[above].sum()) + slots * int(y[tied].sum()) / int(tied.sum())
    return float(hits / k)


def method_metrics(score: np.ndarray, e: np.ndarray, auroc_ceiling: float) -> dict[str, float]:
    """Leaderboard metrics of one ranking against the truth E of one model: AUROC (primary), share of the
    achievable signal, and the two secondary metrics added after the preregistration, AUPRC and precision@61."""
    y = binary_truth(np.asarray(e, dtype=np.float64))
    score = np.asarray(score, dtype=np.float64)
    a = auroc(score, y)
    return {"AUROC": a, "signal_share": signal_share(a, auroc_ceiling), "AUPRC": auprc(score, y),
            "precision_at_61": precision_at_k(score, y)}


def absolute_effect(relative: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
    """E_abs(e): mean over stimuli of ||r_e(b) - r(b)||^2 without dividing by ||r(b)||^2."""
    w = np.asarray(relative, dtype=np.float64)
    r = np.asarray(r_norm, dtype=np.float64)
    if w.shape[1] != len(r):
        raise ValueError("number of stimuli in `relative` differs from `r_norm`")
    return (w * r[None, :]).mean(axis=1)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else float("nan")


def relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| / max(|a|, |b|), 0 where both are zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.maximum(np.abs(a), np.abs(b))
    out = np.zeros_like(m)
    nz = m > 0
    out[nz] = np.abs(a - b)[nz] / m[nz]
    return out


def quantize(x: np.ndarray, digits: int = 6) -> np.ndarray:
    """Round to `digits` significant digits so that float32 summation noise becomes exact ties."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    nz = x != 0
    order = np.floor(np.log10(np.abs(x[nz])))
    scale = 10.0 ** (digits - 1 - order)
    out[nz] = np.round(x[nz] * scale) / scale
    return out
