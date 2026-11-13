"""v1.1 Z1 (threshold sensitivity, added after the study): definitions on synthetic data, consistency on the package."""

import json
from pathlib import Path

import numpy as np
import pytest
import studydata as study_data

from flyvis_circuit_bench import data as fcb_data
from flyvis_circuit_bench.analysis import LEADERBOARD
from flyvis_circuit_bench.metrics import binary_truth
from flyvis_circuit_bench.v11 import z1

ROOT = Path(__file__).resolve().parents[1]
DATA = study_data.DATA


def test_threshold_counts_are_31_61_91_121_for_604_classes():
    assert [z1.threshold_count(604, f) for f in z1.FRACTIONS] == [31, 61, 91, 121]


def test_truth_top_k_matches_binary_truth_and_keeps_ties_and_drops_zeros():
    rng = np.random.default_rng(1)
    e = rng.lognormal(size=604)
    assert (z1.truth_top_k(e, 61) == binary_truth(e)).all()
    tied = np.array([5.0, 4.0, 4.0, 1.0, 0.5])
    assert z1.truth_top_k(tied, 2).sum() == 3  # the tie at the cut is included
    assert z1.truth_top_k(np.array([3.0, 0.0, 0.0, 0.0]), 3).sum() == 1  # zero effect never counts


def _auroc(values: dict[str, list[float]], ks=(10, 20)):
    """auroc[method][k][model] with the first two methods given, the rest copies of the first."""
    models = [f"{i:03d}" for i in range(5)]
    out = {}
    for name in LEADERBOARD:
        v = values.get(name, values["A"])
        out[name] = {k: dict(zip(models, v[ki], strict=True)) for ki, k in enumerate(ks)}
    return out


def test_pair_changes_separates_swaps_from_not_separated():
    plan = tuple(f"{i:03d}" for i in range(5))
    tight = [[0.9, 0.9, 0.9, 0.9, 0.9], [0.8, 0.8, 0.8, 0.8, 0.8]]
    # "P" is clearly above "A" at k=10 and clearly below at k=20 (zero-variance diffs: intervals exclude zero)
    p_vals = [[0.95] * 5, [0.7] * 5]
    res = z1.pair_changes(_auroc({"A": tight, "P": p_vals}), (10, 20), plan)
    # the other methods are copies of A, so every pair with P is a swap; look at the pair (P, A)
    assert ("P", "A") in [(s["a"], s["b"]) for s in res["swaps"]]
    assert res["not_separated"] == []
    # noisy: the sign flips on the means but one interval includes zero -> not separated, not a swap
    noisy = [[0.95, 0.85, 0.9, 0.9, 0.9], [0.9, 0.7, 0.95, 0.75, 0.8]]
    res = z1.pair_changes(_auroc({"A": [[0.9] * 5, [0.8] * 5], "P": noisy}), (10, 20), plan)
    assert res["swaps"] == []
    assert ("P", "A") in [(s["a"], s["b"]) for s in res["not_separated"]]


def test_pair_changes_ignores_pairs_with_a_stable_sign():
    plan = tuple(f"{i:03d}" for i in range(5))
    flat = [[0.8] * 5, [0.8] * 5]
    res = z1.pair_changes(_auroc({"A": flat, "P": [[0.9] * 5, [0.95] * 5]}), (10, 20), plan)
    assert res == {"swaps": [], "not_separated": []}


@pytest.fixture(scope="module")
def block(package):
    return z1.analyze(fcb_data.load(package))


def test_synthetic_block_shape(block):
    n = block["n_classes"]
    assert [t["k"] for t in block["thresholds"]] == [z1.threshold_count(n, f) for f in z1.FRACTIONS]
    assert block["methods"] == list(LEADERBOARD)
    assert len(block["table"]) == len(LEADERBOARD) * 4
    assert all(k <= n for k in block["curve_k"])  # the curve stops at the number of classes
    row = block["table"][0]
    assert set(z1.METRICS) <= set(row) and set(row["AUROC"]) == {"pooled", "original", "replication"}
    assert row["AUROC"]["pooled"]["n"] == 10
    assert block["conclusion"].startswith("Across the four thresholds")


def test_synthetic_patching_is_better_than_synapse_count_at_every_threshold(block):
    by = {(r["method"], r["k"]): r["AUROC"]["pooled"]["mean"] for r in block["table"]}
    for t in block["thresholds"]:
        assert by[("P", t["k"])] > by[("S", t["k"])]


def test_render_has_a_dagger_on_every_table_row_and_is_deterministic(block):
    results = {"v1_1": {"z1": block}}
    text = z1.render_markdown(results)
    assert text == z1.render_markdown(results)
    assert chr(0x2014) not in text and chr(0x2013) not in text
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith(("| method", "| ---")) and "---" not in line:
            assert "†" in line, line
    assert "†" in text.splitlines()[0]
    assert "† †" not in text


def test_figures_are_written_and_reproducible(block, tmp_path):
    results = json.loads(json.dumps({"v1_1": {"z1": block}}))
    a = z1.make_figures(results, tmp_path / "a")
    b = z1.make_figures(results, tmp_path / "b")
    assert sorted(p.name for p in a) == ["v1.1-z1-curve-dark.png", "v1.1-z1-curve-dark.svg",
                                         "v1.1-z1-curve-light.png", "v1.1-z1-curve-light.svg"]
    for p, q in zip(a, b, strict=True):
        assert p.read_bytes() == q.read_bytes()


@pytest.mark.data
@pytest.mark.skipif(not DATA.is_dir(), reason="measurement package not available (set FCB_DATA or run fcb download)")
def test_package_thresholds_and_consistency_with_the_leaderboard():
    r = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    z = r["v1_1"]["z1"]
    assert [t["k"] for t in z["thresholds"]] == [31, 61, 91, 121] and z["n_classes"] == 604
    assert z["curve_k"] == list(range(10, 201, 10))
    lb = {(x["condition"], x["method"]): x for x in r["leaderboard"]["rows"]}
    for row in z["table"]:
        if row["k"] != 61:
            continue
        old = lb[("C1", row["method"])]
        for new_key, old_key in (("AUROC", "AUROC"), ("signal_share", "signal_share"), ("AUPRC", "AUPRC"),
                                 ("precision_at_k", "precision_at_61")):
            assert row[new_key]["pooled"]["mean"] == pytest.approx(old[old_key]["pooled"]["mean"], abs=1e-9)
    assert z["curve"]["P"]["60"]["pooled"]["mean"] > 0.7
    assert all(0 <= v["pooled"]["mean"] <= 1 for m in z["curve"].values() for v in m.values())
    assert "swap" in z["conclusion"]
