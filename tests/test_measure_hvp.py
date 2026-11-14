"""v1.1 Z7: second-order correction of patching on small synthetic networks (no flyvis)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flyvis_circuit_bench.measure import hvp as H  # noqa: E402

HAS_FLYVIS = importlib.util.find_spec("flyvis") is not None
N_CLASSES, T, TYPES, K = 5, 4, 3, 6


def _v(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((K, T, TYPES), generator=g, dtype=torch.float64)


def _quadratic(seed=1, with_quadratic=True):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((T, TYPES), generator=g, dtype=torch.float64)
    b = torch.randn((N_CLASSES, T, TYPES), generator=g, dtype=torch.float64)
    q = torch.randn((N_CLASSES, N_CLASSES, T, TYPES), generator=g, dtype=torch.float64)
    q = 0.5 * (q + q.transpose(0, 1)) * (1.0 if with_quadratic else 0.0)

    def respond(alpha):
        return 3.0 + a + torch.einsum("c,ctu->tu", alpha, b) + 0.5 * torch.einsum("c,d,cdtu->tu", alpha, alpha, q)

    return respond, a, b, q


@pytest.mark.parametrize("backend", H.BACKENDS)
def test_quadratic_function_is_corrected_exactly(backend):
    respond, *_ = _quadratic()
    v = _v()
    res = H.second_order(respond, N_CLASSES, v, backend=backend)
    base = respond(torch.zeros(N_CLASSES, dtype=torch.float64))
    for c in range(N_CLASSES):
        e = torch.zeros(N_CLASSES, dtype=torch.float64)
        e[c] = 1.0
        true = ((respond(e) - base)[None] * v).sum(dim=(1, 2)).numpy()  # the ablation alpha_c = 1
        assert np.allclose(res["first"][:, c] + 0.5 * res["second"][:, c], true, atol=1e-10)
        assert not np.allclose(res["first"][:, c], true, atol=1e-3)  # the first order alone is not exact here


@pytest.mark.parametrize("backend", H.BACKENDS)
def test_linear_function_has_no_correction(backend):
    respond, *_ = _quadratic(with_quadratic=False)
    res = H.second_order(respond, N_CLASSES, _v(), backend=backend)
    assert np.allclose(res["second"], 0.0, atol=1e-10)
    t = H.terms(res)
    assert np.allclose(t["PH"], t["P"]) and np.allclose(t["Q"], 0.0, atol=1e-18)


def _recurrent(seed=3):
    """Unrolled tanh network, the weights of the edges share the scale of their class (as in flyvis)."""
    g = torch.Generator().manual_seed(seed)
    n_nodes = 7
    w = 0.6 * torch.randn((n_nodes, n_nodes), generator=g, dtype=torch.float64)
    cls = torch.randint(0, N_CLASSES, (n_nodes, n_nodes), generator=g)
    x = torch.randn((T, n_nodes), generator=g, dtype=torch.float64)

    def respond(alpha):
        weight = w * (1 - alpha[cls])
        state = torch.zeros(n_nodes, dtype=torch.float64)
        out = []
        for t in range(T):
            state = torch.tanh(weight @ state + x[t])
            out.append(state[:TYPES])
        return torch.stack(out)

    return respond


def test_backends_agree_on_a_recurrent_network():
    respond = _recurrent()
    v = _v(5)
    a = H.second_order(respond, N_CLASSES, v, backend="reverse")
    b = H.second_order(respond, N_CLASSES, v, backend="forward")
    assert np.allclose(a["first"], b["first"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a["second"], b["second"], rtol=1e-9, atol=1e-12)
    assert a["norm"] == pytest.approx(b["norm"])


def test_second_derivative_matches_finite_differences():
    respond = _recurrent()
    v = _v(6)
    res = H.second_order(respond, N_CLASSES, v, backend="forward")
    h = 1e-4
    zero = torch.zeros(N_CLASSES, dtype=torch.float64)
    for c in (0, 3):
        e = torch.zeros(N_CLASSES, dtype=torch.float64)
        e[c] = h
        fd2 = ((respond(e) - 2 * respond(zero) + respond(-e)) / h**2)[None] * v
        assert np.allclose(res["second"][:, c], fd2.sum(dim=(1, 2)).numpy(), rtol=1e-5, atol=1e-6)


def test_classes_subset_leaves_the_rest_nan():
    respond = _recurrent()
    res = H.second_order(respond, N_CLASSES, _v(), classes=[1, 3], backend="forward")
    assert np.isfinite(res["second"][:, [1, 3]]).all() and np.isnan(res["second"][:, [0, 2, 4]]).all()


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        H.second_order(_recurrent(), N_CLASSES, _v(), backend="finite")


@pytest.mark.measure
@pytest.mark.skipif(not HAS_FLYVIS or not os.environ.get("FLYVIS_ROOT_DIR"),
                    reason="needs the [measure] extra and FLYVIS_ROOT_DIR with the flow/0000 ensemble")
def test_flyvis_backends_agree_and_first_order_is_patching():
    from flyvis_circuit_bench.measure import network as N

    net = N.load_trained("000", "cpu")
    movie = 0.5 + 0.2 * torch.rand((12, 1, 721), generator=torch.Generator().manual_seed(0))
    v = torch.from_numpy(N.directions(2, 12, 65, b=0))
    cls = [0, 100, 300]
    a = H.stimulus_terms(net, movie, v, cls, "reverse")
    b = H.stimulus_terms(net, movie, v, cls, "forward")
    for key in ("P", "PH"):
        assert np.allclose(a[key][cls], b[key][cls], rtol=1e-3)
    p = N.patching_terms(net, movie, v).mean(axis=0)
    assert np.allclose(b["P"][cls], p[cls], rtol=1e-3)


def test_preregistered_subset_is_fixed():
    assert H.select_stimuli(40) == [4, 12, 16, 18, 20, 27, 37, 39]
    assert H.select_stimuli(40, 8, 20261101) == H.select_stimuli(40)
    assert H.select_stimuli(40, 8, 1) != H.select_stimuli(40)
    assert len(set(H.select_stimuli(40, 40))) == 40
