"""Statistics shared by every table: mean over models with two 95% confidence intervals.

Every quantity is a mean over models (paired by model id). We always report two intervals because with
N = 5 or N = 10 models the percentile bootstrap undercovers (simulated coverage 0.84 instead of 0.95 at N = 5):

* percentile bootstrap over models (10 000 resamples, fixed seed 20261101), and
* Student t interval (df = N - 1).

The sign rule used for every decision is conservative: a quantity is "positive" only when the lower bounds of
BOTH intervals are > 0, "negative" only when the upper bounds of both are < 0, "inconclusive" otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as st

SEED = 20261101
N_BOOT = 10_000

POSITIVE, NEGATIVE, INCONCLUSIVE, UNDETERMINED = "positive", "negative", "inconclusive", "undetermined"


def bootstrap_ci(values: Sequence[float], alpha: float = 0.05, n_boot: int = N_BOOT,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean (resampling models with replacement)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    lo, hi = np.quantile(x[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def t_ci(values: Sequence[float]) -> tuple[float, float]:
    """Student t interval of the mean, df = N - 1 (nan for a single value)."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return float("nan"), float("nan")
    h = st.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    m = float(x.mean())
    return m - h, m + h


@dataclass(frozen=True)
class Summary:
    """Mean over models with both intervals, per-model values and the models left out (nan or missing)."""

    mean: float
    boot: tuple[float, float]
    t: tuple[float, float]
    per_model: dict[str, float]
    omitted: tuple[str, ...] = field(default=())

    @property
    def min(self) -> float:
        return min(self.per_model.values()) if self.per_model else float("nan")

    @property
    def max(self) -> float:
        return max(self.per_model.values()) if self.per_model else float("nan")

    @property
    def n(self) -> int:
        return len(self.per_model)

    def sign(self) -> str:
        if self.boot[0] > 0 and self.t[0] > 0:
            return POSITIVE
        if self.boot[1] < 0 and self.t[1] < 0:
            return NEGATIVE
        return INCONCLUSIVE

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci_bootstrap": list(self.boot),
            "ci_t": list(self.t),
            "per_model": dict(sorted(self.per_model.items())),
            "min": self.min,
            "max": self.max,
            "n": self.n,
            "omitted": list(self.omitted),
            "sign": self.sign(),
        }


def summarize(per_model: Mapping[str, float], plan: Sequence[str] | None = None) -> Summary:
    """Mean over models with a finite value, in sorted model order (the bootstrap depends on the order).

    `plan`: models the quantity should cover; models absent from `per_model` count as omitted.
    """
    models = sorted(per_model) if plan is None else sorted(plan)
    ok = [m for m in models if m in per_model and np.isfinite(per_model[m])]
    omitted = tuple(m for m in models if m not in ok)
    if not ok:
        nan = float("nan")
        return Summary(nan, (nan, nan), (nan, nan), {}, omitted)
    x = [float(per_model[m]) for m in ok]
    return Summary(float(np.mean(x)), bootstrap_ci(x), t_ci(x), dict(zip(ok, x, strict=True)), omitted)


def one_sided_p(values: Sequence[float], direction: int) -> float:
    """One-sample t test p value for mean > 0 (direction +1) or mean < 0 (direction -1)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    if x.std(ddof=1) == 0:
        return 0.0 if direction * x.mean() > 0 else 1.0
    return float(st.ttest_1samp(x, 0.0, alternative="greater" if direction > 0 else "less").pvalue)


def holm(p: Mapping[str, float], alpha: float = 0.025, m: int | None = None) -> dict[str, bool]:
    """Holm step-down: reject p_(k) <= alpha / (m - k + 1) while all earlier ones were rejected.

    A nan p value (undetermined quantity) counts as 1 and m stays the family size.
    """
    m = len(p) if m is None else m

    def val(k: str) -> float:
        return 1.0 if not math.isfinite(p[k]) else p[k]

    order = sorted(p, key=lambda k: (val(k), k))
    out = {k: False for k in p}
    for i, k in enumerate(order):
        if val(k) <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out
