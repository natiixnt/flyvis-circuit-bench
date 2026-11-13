"""Z4 analysis on a synthetic data dict (the task measurement is not part of the v1.0.0 package)."""

import numpy as np
import pytest

from flyvis_circuit_bench.data import MODELS_POOLED
from flyvis_circuit_bench.v11 import z4

N = 120


def synthetic(correlated: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = rng.lognormal(size=N)
    models = {}
    for m in MODELS_POOLED:
        e = base * rng.lognormal(sigma=0.3, size=N)
        t = e * rng.lognormal(sigma=0.3, size=N) if correlated else rng.lognormal(size=N)
        loss_ref = np.full(16, 1.0)
        loss_abl = loss_ref[None, :] + t[:, None] * rng.uniform(0.8, 1.2, size=(N, 16))
        models[m] = {
            "B_C1": {"E_B": e, "E_B1": e * rng.lognormal(sigma=0.05, size=N),
                     "E_B2": e * rng.lognormal(sigma=0.05, size=N),
                     "S": rng.uniform(10, 100, N), "A": rng.uniform(1, 10, N),
                     "P": e * rng.lognormal(sigma=0.5, size=N),
                     "zrodlo": np.array([f"s{i % 7}" for i in range(N)]),
                     "cel": np.array([f"c{i % 5}" for i in range(N)])},
            "T": {"T": (loss_abl - loss_ref).mean(axis=1), "D": np.abs(t) * rng.lognormal(sigma=0.2, size=N),
                  "loss_ref": loss_ref, "loss_abl": loss_abl, "bodzce": np.array([f"c{i}" for i in range(16)])},
        }
    return {"models": models, "structure": {"n_krawedzi": rng.uniform(5, 50, N)}}


def test_two_by_two_counts_and_off_diagonal_lists():
    e = np.arange(100, 0, -1, dtype=float)
    t = e.copy()
    t[:5], t[50:55] = t[50:55].copy(), t[:5].copy()  # five high-on-E classes swap with five low ones
    tab = z4.two_by_two(e, t, 10)
    assert tab["high_E_low_T"] == 5 and tab["low_E_high_T"] == 5
    assert sum(tab.values()) == 100
    od = z4.off_diagonal(e, t, 10, np.array(["s"] * 100), np.array(["t"] * 100))
    assert [c["class"] for c in od["high_E_low_T"]] == [0, 1, 2, 3, 4]
    assert [c["class"] for c in od["low_E_high_T"]] == [50, 51, 52, 53, 54]


def test_available_requires_all_ten_models():
    d = synthetic()
    assert z4.available(d)
    del d["models"]["007"]["T"]
    assert not z4.available(d)


def test_analyze_correlated_vs_independent_truth_and_conclusion():
    hi = z4.analyze(synthetic(True))
    lo = z4.analyze(synthetic(False))
    assert hi["spearman_E_T"]["pooled"]["mean"] > 0.7 > 0.3 > abs(lo["spearman_E_T"]["pooled"]["mean"])
    assert hi["k"] == 12 and hi["n_clips"] == 16
    assert sum(hi["pooled_mean_table"].values()) == N
    assert hi["conclusion"].startswith("Spearman between the ablation effect")
    assert set(hi["ladder"]) == set(hi["methods"]) and "P" in hi["ladder"]
    assert hi["ladder"]["P"]["AUROC"]["pooled"]["mean"] > lo["ladder"]["P"]["AUROC"]["pooled"]["mean"]


def test_render_has_dagger_and_sections():
    res = {"v1_1": {"z4": z4.analyze(synthetic())}}
    md = z4.render_markdown(res)
    assert md.startswith("# Z4 †") and "Method ladder against T †" in md and "Spearman(D, T)" in md
    assert z4.DAGGER in res["v1_1"]["z4"]["conclusion"] or "†" in md
    assert z4.render_markdown(res) == md  # deterministic


def test_split_half_ceiling_is_one_for_clean_signal_and_lower_with_noise():
    rng = np.random.default_rng(2)
    t = rng.lognormal(size=200)
    ref = np.zeros(16)
    clean = np.repeat(t[:, None], 16, axis=1)
    assert z4.split_half_ceiling(clean, ref, 20) == pytest.approx(1.0)
    noisy = clean * rng.lognormal(sigma=2.0, size=clean.shape)
    assert z4.split_half_ceiling(noisy, ref, 20) < 0.95
