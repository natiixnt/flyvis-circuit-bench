import os
from pathlib import Path

import numpy as np
import pytest
import studydata as study_data

from flyvis_circuit_bench.v11 import z3

ROOT = Path(__file__).resolve().parents[1]
DATA = study_data.DATA


def test_cv_excludes_zero_mean_and_uses_sample_std():
    x = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    cv = z3.cv_across_models(x)
    assert np.isnan(cv[1]) and cv[2] == 0.0
    assert cv[0] == pytest.approx(np.std([1, 3], ddof=1) / 2)


def test_group_bounds_and_groups():
    assert z3.group_bounds(10) == (9, 4)
    assert z3.group_bounds(5) == (5, 2)
    g = z3.groups(np.array([10, 9, 8, 5, 4, 1, 0]), 10)
    assert g["core"].tolist() == [True, True, False, False, False, False, False]
    assert g["variable"].tolist() == [False, False, False, False, True, True, False]
    assert (g["core"] | g["variable"] | g["other"]).all()
    assert not (g["core"] & g["variable"]).any()


def test_pair_matrices_identical_models_and_disjoint_top():
    rng = np.random.default_rng(3)
    a = rng.lognormal(size=200)
    mats = z3.pair_matrices(np.array([a, a, -a + a.max() + 1]), 20)
    assert mats["spearman"][0, 1] == pytest.approx(1.0)
    assert mats["overlap"][0, 1] == 20 and mats["jaccard"][0, 1] == 1.0
    assert mats["overlap"][0, 2] == 0 and mats["jaccard"][0, 2] == 0.0
    assert (mats["spearman"] == mats["spearman"].T).all()
    assert mats["spearman"][0, 2] == pytest.approx(-1.0)


def test_version_block_counts_and_scalar_cv_spearman():
    rng = np.random.default_rng(4)
    e = rng.lognormal(size=(5, 100))
    e[:, 0] = 0.0  # zero effect in every model: excluded from the CV and counted
    scalar = rng.uniform(1, 2, size=(5, 100))
    b = z3.version_block(e, scalar, np.ones(100), 10)
    assert b["n_mean_zero_excluded"] == 1 and b["n_classes_cv_spearman"] == 99
    assert sum(b["group_sizes"].values()) == 100
    assert np.isnan(b["cv_effect"][0])


@pytest.fixture(scope="module")
def results():
    if not (DATA / "SHA256SUMS").is_file():
        if os.environ.get("REQUIRE_STUDY_DATA") == "1":
            pytest.fail(f"study data missing: {DATA}")
        pytest.skip("study data missing")
    from flyvis_circuit_bench import data

    return {"v1_1": {"z3": z3.analyze(data.load(DATA))}}


def test_package_matrix_shape_and_diagonal(results):
    z = results["v1_1"]["z3"]
    m = np.array(z["spearman_matrix"])
    assert m.shape == (10, 10) and np.allclose(np.diag(m), 1) and np.allclose(m, m.T)
    o = np.array(z["overlap_matrix"])
    assert (np.diag(o) >= 61).all() and z["k"] == 61
    assert z["pairs"]["pooled"]["spearman"]["n"] == 45


def test_package_groups_and_conclusion(results):
    z = results["v1_1"]["z3"]
    for v, n in (("pooled", 10), ("original", 5), ("replication", 5)):
        b = z["blocks"][v]
        assert b["n_models"] == n and sum(b["group_sizes"].values()) == 604
    assert "†" in z3.render_markdown(results)
    assert z["conclusion"] in z3.render_markdown(results)
    assert z["pairs"]["pooled"]["spearman"]["median"] < z["split_half_spearman"]["pooled"]["mean"]


def test_cross_model_auroc_synthetic():
    rng = np.random.default_rng(5)
    base = rng.lognormal(size=(4, 150))
    e = np.vstack([base[0], base[0], base[0], base[0]])  # identical models: A predicts B perfectly
    s = np.ones((4, 150))  # constant anatomy: AUROC 0.5
    r = z3.cross_model_auroc(e, s, 15)
    assert r["model_a_e"]["n"] == 12 and r["model_a_e"]["min"] == 1.0
    assert r["synapse_count"]["median"] == 0.5 and r["n_pairs_a_better"] == 12


def test_package_cross_model_block(results):
    cm = results["v1_1"]["z3"]["cross_model_auroc"]
    assert cm["model_a_e"]["n"] == 90 and cm["synapse_count"]["n"] == 10 and cm["n_pairs"] == 90
    assert "added after Z3 results" in z3.render_markdown(results)
