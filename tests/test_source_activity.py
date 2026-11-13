"""The two methods without learned weights (source activity only, synapse count x source activity) and the names."""

from pathlib import Path

import numpy as np
import pytest
import studydata as study_data

from flyvis_circuit_bench import analysis as A
from flyvis_circuit_bench import data as D
from flyvis_circuit_bench import names as N
from flyvis_circuit_bench.source_activity import TYPE_INDEX, source_activity, synapses_x_activity

ROOT = Path(__file__).resolve().parents[1]
DATA = study_data.DATA


def test_source_activity_is_the_activity_of_the_source_type():
    a = np.arange(65, dtype=float) + 0.5
    out = source_activity(["R1", "L2", "R1", "TmY18"], a)
    assert out.tolist() == [0.5, 9.5, 0.5, 64.5]
    assert synapses_x_activity([2, 3, 4, 5], ["R1", "L2", "R1", "TmY18"], a).tolist() == [1.0, 28.5, 2.0, 322.5]


def test_type_index_is_a_one_to_one_map_into_65_types():
    assert len(set(TYPE_INDEX.values())) == len(TYPE_INDEX) == 63
    assert min(TYPE_INDEX.values()) == 0 and max(TYPE_INDEX.values()) == 64


def test_scores_come_from_activity_and_synapses_only(package):
    """Scrambling every learned-weight score (A, M_akt, M5, M3, M1) leaves both new rows unchanged."""
    data = D.load(package)
    d = data["models"]["000"]
    before = A.c1_scores(d, None)
    m5 = d["M5"]
    expected = source_activity(m5["zrodlo"], m5["a_typu"])
    assert np.array_equal(before["act"], expected)
    assert np.array_equal(before["S_act"], d["B_C1"]["S"] * expected)
    rng = np.random.default_rng(0)
    for kind, keys in (("B_C1", ("A",)), ("M5", ("M_akt", "M5")), ("M13", ("M1", "M3"))):
        for k in keys:
            d[kind][k] = rng.random(d[kind][k].shape)
    after = A.c1_scores(d, None)
    assert np.array_equal(after["act"], before["act"]) and np.array_equal(after["S_act"], before["S_act"])
    assert not np.array_equal(after["M_akt"], before["M_akt"])


def test_changing_the_activity_changes_the_rows(package):
    d = D.load(package)["models"]["000"]
    before = A.c1_scores(d, None)["S_act"].copy()
    d["M5"]["a_typu"] = d["M5"]["a_typu"][::-1].copy()
    assert not np.array_equal(A.c1_scores(d, None)["S_act"], before)


def test_leaderboard_has_the_new_rows(results):
    keys = {x["key"] for x in results["leaderboard"]["rows"]}
    assert {"C1|act", "C1|S_act"} <= keys


def test_no_method_with_weights_is_called_plain_activity():
    """Every method that multiplies by learned weights says so in its name; plain activity is a different row."""
    for key in ("M_akt", "M3", "M5"):
        assert "weights" in N.METHODS[key], key
        assert "trained weights" in N.COST[key], key
        assert N.ACCESS_OF[key] == "trained weights + forward pass"
    assert N.METHODS["act"] == "source activity only †" and "weights" not in N.METHODS["S_act"]
    for key in ("act", "S_act"):
        assert "no learned weights" in N.COST[key] and N.ACCESS_OF[key] == "forward pass, no learned weights"
    assert set(N.ACCESS_OF.values()) == set(N.ACCESS)
    assert len(set(N.METHODS.values())) == len(N.METHODS) - 2  # P_F2 and M1_F2 share a name with P and M1


@pytest.mark.data
@pytest.mark.skipif(not DATA.is_dir(), reason="measurement package not available (set FCB_DATA or run fcb download)")
def test_type_index_reproduces_the_stored_weighted_activity_in_all_models():
    """M_akt = sum of |weights| x a_typu[source type]: the check that the table indexes a_typu correctly."""
    for m in D.MODELS_POOLED:
        s = D.load_npz(DATA / "activity_sae" / f"C1_{m}.npz")
        w = D.load_npz(DATA / "ablation" / f"C1_{m}.npz")["A"]
        # float32 weights were used when M_akt was stored, hence the tolerance
        assert np.allclose(w * source_activity(s["zrodlo"], s["a_typu"]), s["M_akt"], rtol=1e-6)
