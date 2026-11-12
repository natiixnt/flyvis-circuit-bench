import math

import numpy as np
import pytest

from flyvis_circuit_bench import metrics as M
from flyvis_circuit_bench import stats as S


def test_binary_truth_top10_with_ties_and_zeros():
    e = np.arange(100, dtype=float)
    assert M.binary_truth(e).sum() == 10
    e[89] = e[90]  # tie at the threshold is included
    assert M.binary_truth(e).sum() == 11
    z = np.zeros(100)
    z[:3] = 1.0  # 10th largest is 0: only classes with an effect count
    assert M.binary_truth(z).sum() == 3


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    s = rng.normal(size=50).round(1)  # rounding creates ties
    y = rng.random(50) < 0.3
    pos, neg = s[y], s[~y]
    pairwise = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert M.auroc(s, y) == pytest.approx(pairwise)
    assert math.isnan(M.auroc(s, np.zeros(50, bool)))


def test_auprc_perfect_and_worst():
    y = np.array([1, 1, 0, 0, 0], bool)
    assert M.auprc(np.array([5, 4, 3, 2, 1.0]), y) == pytest.approx(1.0)
    assert M.auprc(np.array([1, 2, 3, 4, 5.0]), y) == pytest.approx((1 / 4 + 2 / 5) / 2)


def test_c1_is_nan_when_degenerate():
    e = np.arange(100, dtype=float)
    assert M.spearman_positives(e, e) == pytest.approx(1.0)
    assert math.isnan(M.spearman_positives(np.ones(100), e))
    with pytest.raises(ValueError):
        M.measure("nope", e, e)


def test_ceiling_fraction_and_absolute_effect():
    assert M.ceiling_fraction(0.75, 1.0) == pytest.approx(0.5)
    assert math.isnan(M.ceiling_fraction(0.9, 0.5))
    rel = np.array([[1.0, 2.0], [0.0, 4.0]])
    assert M.absolute_effect(rel, np.array([10.0, 1.0])).tolist() == [6.0, 2.0]
    with pytest.raises(ValueError):
        M.absolute_effect(rel, np.ones(3))


def test_quantize_makes_float32_noise_a_tie():
    a = np.array([0.01 * 3, 0.01 * 3 * (1 + 2e-7), 0.0])
    q = M.quantize(a)
    assert q[0] == q[1] and q[2] == 0.0


def test_bootstrap_is_deterministic_and_t_interval_known():
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert S.bootstrap_ci(x) == S.bootstrap_ci(x)
    lo, hi = S.t_ci(x)
    assert (lo + hi) / 2 == pytest.approx(0.3)
    assert hi - lo == pytest.approx(2 * 2.7764451 * np.std(x, ddof=1) / math.sqrt(5), rel=1e-6)


def test_summarize_omits_nan_and_missing_models_in_sorted_order():
    s = S.summarize({"002": 0.3, "000": 0.1, "001": float("nan")}, plan=("000", "001", "002", "003"))
    assert list(s.per_model) == ["000", "002"]
    assert s.omitted == ("001", "003")
    assert s.mean == pytest.approx(0.2)


def test_sign_rule_needs_both_intervals():
    pos = S.summarize({f"{i:03d}": v for i, v in enumerate([0.2, 0.21, 0.19, 0.22, 0.2])})
    assert pos.sign() == S.POSITIVE
    mixed = S.summarize({f"{i:03d}": v for i, v in enumerate([0.3, -0.1, 0.2, 0.25, 0.05])})
    assert mixed.sign() == S.INCONCLUSIVE
    neg = S.summarize({f"{i:03d}": -v for i, v in enumerate([0.2, 0.21, 0.19, 0.22, 0.2])})
    assert neg.sign() == S.NEGATIVE


def test_holm_step_down_and_nan_as_one():
    p = {"a": 0.001, "b": 0.008, "c": 0.02, "d": float("nan")}
    assert S.holm(p, alpha=0.025, m=4) == {"a": True, "b": True, "c": False, "d": False}
    assert S.holm({"a": 0.03, "b": 0.001}, alpha=0.025) == {"a": False, "b": True}


def test_one_sided_p_direction():
    x = [0.2, 0.25, 0.22, 0.19, 0.3]
    assert S.one_sided_p(x, 1) < 0.001 < S.one_sided_p(x, -1)
    assert S.one_sided_p([0.1, 0.1], 1) == 0.0
    assert math.isnan(S.one_sided_p([0.1], 1))
