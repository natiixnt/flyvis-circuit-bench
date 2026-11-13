"""Step `task`: the pure helpers on a synthetic network (no flyvis, no torch); the flyvis run is marked `measure`."""

import contextlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from flyvis_circuit_bench.measure import task as TK

HAS_FLYVIS = importlib.util.find_spec("flyvis") is not None and importlib.util.find_spec("torch") is not None


class FakeNet:
    """Decoded flow = sum over classes of scale[c] * basis[c]; zeroing a class removes its contribution."""

    def __init__(self, n_classes=12, shape=(4, 6, 2, 9), seed=0):
        rng = np.random.default_rng(seed)
        self.basis = rng.normal(size=(n_classes, *shape))
        self.scale = np.ones(n_classes)
        self.calls = 0

    def decode(self):
        self.calls += 1
        return np.tensordot(self.scale, self.basis, axes=1)

    @contextlib.contextmanager
    def ablate(self, e):
        old = self.scale[e]
        self.scale[e] = 0.0
        try:
            yield
        finally:
            self.scale[e] = old


def test_clip_losses_is_flyvis_l2norm():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(5, 3, 2, 7)), rng.normal(size=(5, 3, 2, 7))
    flyvis_l2norm = np.sqrt(((a - b) ** 2).sum(axis=(1, 2, 3))).mean()  # objectives.l2norm
    assert TK.clip_losses(a, b).mean() == pytest.approx(flyvis_l2norm)
    with pytest.raises(ValueError):
        TK.clip_losses(a, b[:4])


def test_flow_change_definition_and_zero_reference():
    ref = np.ones((2, 3, 2, 4))
    assert TK.flow_change(ref, ref).tolist() == [0.0, 0.0]
    assert TK.flow_change(np.zeros_like(ref), ref) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        TK.flow_change(ref, np.zeros_like(ref))


def test_task_effects_on_synthetic_network_is_deterministic_and_exact():
    net = FakeNet()
    gt = net.decode() + 0.1  # the intact network is close to the target
    r1 = TK.task_effects(net.decode, net.ablate, 12, gt)
    r2 = TK.task_effects(net.decode, net.ablate, 12, gt)
    assert np.array_equal(r1["T"], r2["T"]) and np.array_equal(r1["D"], r2["D"])
    assert (net.scale == 1).all()  # every ablation was restored
    e = 3
    y = net.decode() - net.basis[e]
    expected = (TK.clip_losses(y, gt) - TK.clip_losses(net.decode(), gt)).mean()
    assert r1["T"][e] == pytest.approx(expected)
    assert r1["loss_abl"].shape == (12, 4) and r1["change"].shape == (12, 4)


def test_task_effects_subset_classes_matches_full_run_and_leaves_nan():
    net = FakeNet()
    gt = np.zeros_like(net.decode())
    full = TK.task_effects(net.decode, net.ablate, 12, gt)
    sub = TK.task_effects(net.decode, net.ablate, 12, gt, classes=[2, 5, 7, 9, 11])
    done = [2, 5, 7, 9, 11]
    assert np.array_equal(sub["T"][done], full["T"][done])  # 5 classes computed twice give the same T
    assert np.isnan(np.delete(sub["T"], done)).all()


def test_sink_receives_intact_and_every_measured_ablation():
    net = FakeNet(n_classes=4)
    seen = []
    TK.task_effects(net.decode, net.ablate, 4, np.zeros_like(net.decode()), classes=[1, 3],
                    sink=lambda e, y: seen.append(e))
    assert seen == [-1, 1, 3]


@pytest.mark.measure
@pytest.mark.skipif(not HAS_FLYVIS, reason="needs the [measure] extra and flyvis data")
def test_flyvis_task_five_classes_twice_gives_the_same_T(tmp_path):
    stimuli = Path(os.environ.get("FCB_STIMULI", "data/measurements-v1.1.0/stimuli_B.json"))
    flow = os.environ.get("FCB_SINTEL_FLOW")
    if not stimuli.is_file() or not flow:
        pytest.skip("needs stimuli_B.json and FCB_SINTEL_FLOW (MPI Sintel training/flow)")
    from flyvis_circuit_bench.measure import run

    classes = [0, 100, 250, 400, 603]
    a = run.measure("task", "C1", "000", stimuli, tmp_path / "a", flow_dir=Path(flow), classes=classes,
                    store_flow="none")[0]
    b = run.measure("task", "C1", "000", stimuli, tmp_path / "b", flow_dir=Path(flow), classes=classes,
                    store_flow="none")[0]
    ta, tb = np.load(a)["T"], np.load(b)["T"]
    assert np.array_equal(ta[classes], tb[classes]) and np.isfinite(ta[classes]).all()
