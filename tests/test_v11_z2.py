"""Z2 (v1.1 †): truth split by stimulus family. Unit tests on a synthetic package, conformity on the real one."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import studydata

from flyvis_circuit_bench import data as D
from flyvis_circuit_bench import metrics as M
from flyvis_circuit_bench.v11 import z2

N, N_EDGES, N_SINTEL = 120, 24, 16
NAMES = np.array([f"krawedz_k{i:03d}" for i in range(N_EDGES)] + [f"sintel_clip_{i}" for i in range(N_SINTEL)])


def synthetic(noise: float = 0.0, seed: int = 3) -> dict:
    """Per-stimulus effects = shared class effect x stimulus gain (+ noise); every method is the class effect."""
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(0, 1.5, N))
    rel = base[:, None] * np.exp(rng.normal(0, noise, (N, len(NAMES))))
    b = {"wzgledne": rel, "bodzce": NAMES, "E_B": rel.mean(axis=1), "E_B1": rel.mean(axis=1),
         "E_B2": rel.mean(axis=1), "S": base, "A": base, "P": base}
    models = {m: {"B_C1": b} for m in D.MODELS_POOLED}
    return {"models": models, "structure": None, "k5": None}


def test_family_index_and_unknown_names():
    idx = z2.family_index(NAMES)
    assert len(idx["edges"]) == N_EDGES and len(idx["sintel"]) == N_SINTEL
    with pytest.raises(ValueError):
        z2.family_index(np.array(["stim_00", "krawedz_a"]))


def test_family_effects_are_means_over_the_family():
    rel = np.arange(N * len(NAMES), dtype=float).reshape(N, -1)
    e = z2.family_effects(rel, NAMES)
    assert np.allclose(e["edges"], rel[:, :N_EDGES].mean(axis=1))
    assert np.allclose(e["sintel"], rel[:, N_EDGES:].mean(axis=1))


def test_identical_families_agree_perfectly():
    r = z2.analyze(synthetic(noise=0.0))
    assert r["possible"] and r["consistent_with_E_B"]
    assert r["agreement"]["spearman"]["pooled"]["mean"] == pytest.approx(1.0)
    assert r["agreement"]["top61_jaccard"]["pooled"]["mean"] == pytest.approx(1.0)
    assert r["ceiling_spearman"]["edges"]["pooled"]["mean"] == pytest.approx(1.0)
    assert r["ladder"]["P"]["edges"]["AUROC"]["pooled"]["mean"] == pytest.approx(1.0)
    assert r["ladder"]["P"]["edges"]["signal_share"]["pooled"]["mean"] == pytest.approx(1.0)
    assert r["n_positive"]["edges"] == [int(np.ceil(0.1 * N))]


def test_noise_lowers_agreement_and_ceiling():
    clean, noisy = z2.analyze(synthetic(0.0)), z2.analyze(synthetic(1.0))
    assert noisy["agreement"]["spearman"]["pooled"]["mean"] < clean["agreement"]["spearman"]["pooled"]["mean"]
    assert noisy["ceiling_spearman"]["sintel"]["pooled"]["mean"] < 1.0


def test_split_half_is_deterministic_and_seeded_per_model_and_family():
    rel = synthetic(0.8)["models"]["000"]["B_C1"]["wzgledne"][:, :N_EDGES]
    a, b = z2.split_half(rel, "000", "edges"), z2.split_half(rel, "000", "edges")
    assert a == b
    assert z2.split_half(rel, "001", "edges") != a
    assert z2.split_half(rel, "000", "sintel") != a


def test_top61_overlap_counts_the_intersection():
    d = synthetic(0.8)
    r = z2.model_result(d["models"]["000"], None, "000")
    e = z2.family_effects(d["models"]["000"]["B_C1"]["wzgledne"], NAMES)
    ta, tb = M.binary_truth(e["edges"]), M.binary_truth(e["sintel"])
    assert r["agreement"]["top61_overlap"] == int((ta & tb).sum())


def test_not_possible_without_per_stimulus_effects_or_family_names():
    d = synthetic()
    del d["models"]["000"]["B_C1"]["wzgledne"]
    d["models"] = {"000": d["models"]["000"]}
    assert z2.analyze(d) == {"possible": False,
                             "reason": "per-stimulus effects are not stored in the measurement package"}
    d2 = synthetic()
    d2["models"]["000"]["B_C1"]["bodzce"] = np.array([f"stim_{i}" for i in range(len(NAMES))])
    assert z2.analyze(d2)["possible"] is False
    assert "not possible" in z2.render(z2.analyze(d2)).lower()


def test_render_has_dagger_and_escapes_pipes():
    text = z2.render(z2.analyze(synthetic(0.5)))
    assert "†" in text and "\\|" in text


# ---------------------------------------------------------------- conformity on the real package

def _package() -> Path | None:
    return studydata.DATA if (studydata.DATA / "SHA256SUMS").is_file() else None


def test_real_package_family_split_consistent_with_stored_truth():
    pkg = _package()
    if pkg is None:
        if os.environ.get("REQUIRE_STUDY_DATA") == "1":
            pytest.fail("the measurement package 1.1.0 is required (FCB_DATA or data/measurements-v1.1.0)")
        pytest.skip("measurement package not available (set FCB_DATA)")
    data = D.load(pkg)
    r = z2.analyze(data)
    assert r["possible"] and r["consistent_with_E_B"]
    assert r["n_stimuli"] == {"edges": 24, "sintel": 16}
    assert r["agreement"]["spearman"]["pooled"]["n"] == 10
    assert r["agreement"]["spearman"]["pooled"]["mean"] > 0.9
    assert set(z2.analyze(data)["ladder"]) == set(z2.LEADERBOARD)
