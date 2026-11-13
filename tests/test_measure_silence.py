"""Step `silence`: the pure driver on a synthetic network (no flyvis, no torch)."""

import contextlib

import numpy as np
import pytest

from flyvis_circuit_bench.measure import silence as SL


class FakeNet:
    """Response of stimulus b = sum over classes of scale[c] * basis[b, c]; silencing zeroes classes."""

    def __init__(self, n_classes=9, n_stim=4, seed=0):
        rng = np.random.default_rng(seed)
        self.basis = rng.uniform(0.5, 1.5, size=(n_stim, n_classes, 6, 3))
        self.scale = np.ones(n_classes)

    def respond(self):
        return [np.tensordot(self.scale, self.basis[b], axes=1) for b in range(len(self.basis))]

    @contextlib.contextmanager
    def silence(self, idx):
        old = self.scale[idx].copy()
        self.scale[idx] = 0.0
        try:
            yield
        finally:
            self.scale[idx] = old


SOURCES = ["B", "A", "A", "C", "B", "A", "C", "C", "C"]


def test_type_groups_cover_every_class_once_and_sort_by_name():
    g = SL.type_groups(SOURCES)
    assert [t for t, _ in g] == ["A", "B", "C"]
    assert sorted(i for _, idx in g for i in idx) == list(range(9))
    assert dict(g)["A"] == [1, 2, 5]


def test_silence_effects_match_definition_and_restore_scales():
    net = FakeNet()
    groups = SL.type_groups(SOURCES)
    rel = SL.silence_effects(net.respond, net.silence, groups)
    assert (net.scale == 1).all() and rel.shape == (3, 4)
    ref = net.respond()
    idx = dict(groups)["A"]
    ablated = [r - np.tensordot(np.ones(3), net.basis[b][idx], axes=1) for b, r in enumerate(ref)]
    expected = [np.sum((a - r) ** 2) / np.sum(r**2) for a, r in zip(ablated, ref, strict=True)]
    assert rel[0] == pytest.approx(expected)


def test_subset_of_types_and_repeatability():
    net = FakeNet()
    groups = SL.type_groups(SOURCES)
    a = SL.silence_effects(net.respond, net.silence, groups, types=["C"])
    b = SL.silence_effects(net.respond, net.silence, groups, types=["C"])
    assert np.isnan(a[:2]).all() and np.array_equal(a[2], b[2])
    full = SL.silence_effects(net.respond, net.silence, groups)
    assert np.array_equal(full[2], a[2])


def test_nonfinite_response_gives_nan():
    net = FakeNet()
    calls = {"n": 0}

    def respond():
        calls["n"] += 1
        r = net.respond()
        if calls["n"] == 2:  # the first silenced run
            r[0][0, 0] = np.inf
        return r

    rel = SL.silence_effects(respond, net.silence, SL.type_groups(SOURCES))
    assert np.isnan(rel[0]).all() and np.isfinite(rel[1:]).all()


@pytest.mark.parametrize("what", ["ablation", "cheap", "sae", "structure", "task", "all"])
def test_types_option_is_rejected_for_every_step_but_silence(what, tmp_path):
    from flyvis_circuit_bench import cli
    from flyvis_circuit_bench.measure import run

    with pytest.raises(ValueError, match="silence step only"):
        run.measure(what, "C1", "000", tmp_path / "s.json", tmp_path, types=["A"])
    argv = ["measure", "--what", what, "--condition", "trained", "--model", "000", "--stimuli",
            str(tmp_path / "s.json"), "--out", str(tmp_path), "--types", "A"]
    assert cli.main(argv) == 2


@pytest.mark.parametrize("what", ["ablation", "cheap", "sae", "structure", "silence", "all"])
def test_classes_option_is_rejected_for_every_step_but_task(what, tmp_path):
    from flyvis_circuit_bench.measure import run

    with pytest.raises(ValueError, match="task step only"):
        run.measure(what, "C1", "000", tmp_path / "s.json", tmp_path, classes=[1])


def test_limit_vram_is_a_noop_on_the_cpu():
    from flyvis_circuit_bench.measure import network as N

    N.limit_vram("cpu")
    assert N.VRAM_LIMIT_MIB == 3300 and N.VRAM_FREE_MIB == 3584
