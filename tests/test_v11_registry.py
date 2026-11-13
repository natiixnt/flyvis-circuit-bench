"""Registry of the v1.1 analyses: the interface of every module and the order of results.json."""

import json
from pathlib import Path

import pytest
import studydata as study_data

from flyvis_circuit_bench import v11

ROOT = Path(__file__).resolve().parents[1]
DATA = study_data.DATA


def test_every_module_exposes_the_interface_and_a_unique_key():
    mods = v11.modules()
    assert [m.KEY for m in mods] == list(v11.MODULES)
    assert len({m.KEY for m in mods}) == len(mods)
    for m in mods:
        for attr in ("analyze", "render_markdown", "make_figures"):
            assert callable(getattr(m, attr)), (m.KEY, attr)


def test_outputs_and_figures_only_for_keys_present():
    assert v11.outputs({"v1_1": {}}, Path("docs")) == {}
    assert v11.make_figures({"v1_1": {}}, Path("unused")) == []


def test_published_results_follow_the_registry_order():
    path = ROOT / "results" / "results.json"
    if not path.is_file():
        pytest.skip("no results.json")
    keys = list(json.loads(path.read_text(encoding="utf-8"))["v1_1"])
    assert keys == [k for k in v11.MODULES if k in keys] and {"z1", "z2", "z3"} <= set(keys)


def test_analysis_without_its_data_is_skipped():
    from flyvis_circuit_bench.v11 import z4

    assert not z4.available({"models": {}})
