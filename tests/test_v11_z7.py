"""Z7 analysis on a synthetic data dict (the hvp measurement is not part of the v1.0.0 package)."""

import numpy as np
import pytest

from flyvis_circuit_bench.data import MODELS_POOLED
from flyvis_circuit_bench.v11 import z7

N = 120


def synthetic(gain: float = 0.0, seed: int = 0) -> dict:
    """`gain` > 0: the corrected score follows the truth more closely than patching."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(size=N)
    models = {}
    for m in MODELS_POOLED:
        e = base * rng.lognormal(sigma=0.3, size=N)
        noise = rng.lognormal(sigma=1.0, size=N)
        p = e * noise
        ph = e * noise ** (1.0 - gain)
        models[m] = {
            "B_C1": {"E_B": e, "E_B1": e * rng.lognormal(sigma=0.05, size=N),
                     "E_B2": e * rng.lognormal(sigma=0.05, size=N), "P": p},
            "HVP": {"PH": ph, "P_chk": p * (1 + 1e-6 * rng.standard_normal(N)), "cross": 0.1 * ph, "Q": 0.01 * ph,
                    "meta": {"backend": "forward", "K": 32, "n_stimuli": 8, "stimuli_index": [4, 12]}},
        }
    return {"models": models}


def test_available_requires_all_ten_models():
    d = synthetic()
    assert z7.available(d)
    del d["models"]["004"]["HVP"]
    assert not z7.available(d)


def test_gap_closed_formula():
    assert z7.gap_closed(0.97, 0.985) == pytest.approx(0.5)
    assert z7.gap_closed(0.9, 0.9) == 0.0
    assert np.isnan(z7.gap_closed(1.0, 1.0))


def test_analyze_detects_a_better_corrected_score():
    better = z7.analyze(synthetic(gain=1.0))
    same = z7.analyze(synthetic(gain=0.0))
    assert better["diff"]["AUROC"]["pooled"]["mean"] > 0.02
    assert better["gap_closed"]["pooled"] > 0.2
    assert abs(same["diff"]["AUROC"]["pooled"]["mean"]) < 1e-9 and abs(same["gap_closed"]["pooled"]) < 1e-9
    assert better["k"] == 12 and set(better["methods"]) == {"P", "P+HVP", "P40"} and better["n_stimuli"] == 8
    assert better["check_P_spearman"]["pooled"]["mean"] > 0.999
    assert better["meta"]["000"] == {"backend": "forward", "K": 32, "n_stimuli": 8, "stimuli_index": [4, 12]}


def test_analyze_is_deterministic_and_changes_with_the_data():
    a = z7.analyze(synthetic(1.0))
    assert a == z7.analyze(synthetic(1.0))
    assert a["methods"]["P+HVP"]["AUROC"]["pooled"]["mean"] != z7.analyze(synthetic(0.5))["methods"]["P+HVP"][
        "AUROC"]["pooled"]["mean"]


def test_render_has_dagger_rows_and_both_intervals():
    res = {"v1_1": {"z7": z7.analyze(synthetic(1.0))}}
    md = z7.render_markdown(res)
    assert md.startswith("# Z7 †") and "Patching + Hessian correction †" in md and "t interval" in md
    assert "Patching, same 8 stimuli" in md and "(context)" in md
    assert "Gap to the ceiling closed" in md and z7.render_markdown(res) == md
    assert z7.make_figures(res, "unused") == []


# ---------------------------------------------------------------- conformity on the real package 1.1.1


def test_real_package_reproduces_the_committed_z7_block():
    import json
    import os
    from pathlib import Path

    import studydata

    from flyvis_circuit_bench import data as D
    from flyvis_circuit_bench.report import to_json

    pkg = studydata.DATA
    have = (pkg / "SHA256SUMS").is_file() and all((pkg / rel).is_file() for rel in D.v111_files())
    if not have:
        if os.environ.get("REQUIRE_STUDY_DATA") == "1":
            pytest.fail("the measurement package 1.1.1 is required (FCB_DATA or data/measurements-v1.1.1)")
        pytest.skip("package 1.1.1 (hvp/) not available")
    committed = json.loads((Path(__file__).resolve().parents[1] / "results" / "results.json")
                           .read_text(encoding="utf-8"))["v1_1"]["z7"]
    loaded = D.load(pkg)
    assert z7.available(loaded)
    got = z7.analyze(loaded)
    assert to_json({"x": got}) == to_json({"x": committed})
    assert got["n_stimuli"] == 8 and got["k"] == 61
    # a mutation of one stored score changes the result (the test is not vacuous)
    loaded["models"]["000"]["HVP"]["PH"] = loaded["models"]["000"]["HVP"]["PH"][::-1].copy()
    assert to_json({"x": z7.analyze(loaded)}) != to_json({"x": committed})
