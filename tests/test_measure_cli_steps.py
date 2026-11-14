"""The `fcb measure` options reach `run.measure` unchanged for every step (old steps work after hvp)."""

import json

import numpy as np
import pytest

from flyvis_circuit_bench import cli, data
from flyvis_circuit_bench.measure import run
from flyvis_circuit_bench.v11 import z7


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    calls = []

    def fake(what, condition, model, stimuli, out, **kwargs):
        calls.append((what, condition, model, kwargs))
        return [tmp_path / f"{what}.npz"]

    monkeypatch.setattr(run, "measure", fake)
    return calls


def _argv(tmp_path, *extra):
    return ["measure", "--condition", "trained", "--model", "003", "--stimuli", str(tmp_path / "s.json"),
            "--out", str(tmp_path), *extra]


def test_task_step_gets_store_flow_flow_dir_and_classes(recorded, tmp_path, capsys):
    assert cli.main(_argv(tmp_path, "--what", "task", "--flow-dir", str(tmp_path / "flow"), "--store-flow", "none",
                          "--classes", "1,5")) == 0
    what, condition, model, kw = recorded[0]
    assert (what, condition, model) == ("task", "C1", "003")
    assert kw["store_flow"] == "none" and kw["classes"] == [1, 5] and kw["flow_dir"] == tmp_path / "flow"
    assert kw["backend"] == "forward" and kw["n_stimuli"] is None and kw["max_stimuli"] is None


def test_silence_step_gets_types(recorded, tmp_path):
    assert cli.main(_argv(tmp_path, "--what", "silence", "--types", "Mi1,T4a")) == 0
    kw = recorded[0][3]
    assert kw["types"] == ["Mi1", "T4a"] and kw["store_flow"] == "float16" and kw["classes"] is None


def test_cheap_step_defaults_are_unchanged(recorded, tmp_path):
    assert cli.main(_argv(tmp_path, "--what", "cheap")) == 0
    what, _, _, kw = recorded[0]
    assert what == "cheap" and kw["k"] == 32 and kw["types"] is None and kw["classes"] is None


def test_hvp_step_gets_its_options(recorded, tmp_path):
    assert cli.main(_argv(tmp_path, "--what", "hvp", "--n-stimuli", "8", "--backend", "reverse",
                          "--classes", "0,100", "--device", "cuda")) == 0
    kw = recorded[0][3]
    assert kw["n_stimuli"] == 8 and kw["backend"] == "reverse" and kw["classes"] == [0, 100]


def test_hvp_step_is_for_trained_networks_only(tmp_path):
    with pytest.raises(ValueError, match="trained networks only"):
        run.measure("hvp", "C3", "000", tmp_path / "s.json", tmp_path)


def test_hvp_meta_written_by_the_measurement_is_read_back_as_a_dict(tmp_path):
    """measure_hvp saves `meta=json.dumps(...)`; data.load turns it into a dict that z7._meta reads."""
    meta = {"condition": "C1", "model": "000", "device": "cuda", "K": 32, "backend": "forward", "n_stimuli": 8,
            "stimuli_index": [4, 12, 16, 18, 20, 27, 37, 39], "max_allocated_mib": 1429.2}
    path = tmp_path / "hvp" / "C1_000.npz"
    path.parent.mkdir()
    np.savez(path, PH=np.ones(5), P_chk=np.ones(5), cross=np.zeros(5), Q=np.zeros(5), meta=json.dumps(meta))
    loaded = data.load_npz(path)
    assert loaded["meta"] == meta
    assert z7._meta(loaded) == {"backend": "forward", "K": 32, "n_stimuli": 8,
                                "stimuli_index": [4, 12, 16, 18, 20, 27, 37, 39]}
    assert z7._meta({}) == {"backend": None, "K": None, "n_stimuli": None, "stimuli_index": None}
