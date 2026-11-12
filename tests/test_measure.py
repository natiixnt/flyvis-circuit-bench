"""Measurement path: pure helpers always; a flyvis smoke test only with the [measure] extra installed."""

import importlib.util
import os

import numpy as np
import pytest

from flyvis_circuit_bench.measure import network as N

HAS_FLYVIS = importlib.util.find_spec("flyvis") is not None and importlib.util.find_spec("torch") is not None


def test_ablation_effect_is_relative_squared_change():
    ref = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    abl = [np.array([[1.0, 0.0], [0.0, 0.0]])]
    assert N.ablation_effect(ref, abl).tolist() == [0.5]
    with pytest.raises(ValueError):
        N.ablation_effect(ref, [np.zeros((3, 2))])
    with pytest.raises(ValueError):
        N.ablation_effect([np.zeros((2, 2))], [np.zeros((2, 2))])


def test_directions_fixed_seed_and_prefix_property():
    a = N.directions(32, 10, 65, b=3)
    b = N.directions(64, 10, 65, b=3)
    assert a.dtype == np.float32 and a.shape == (32, 10, 65)
    assert np.array_equal(a, b[:32])
    assert not np.array_equal(a, N.directions(32, 10, 65, b=4))


def test_effects_and_groups():
    rel = np.arange(12, dtype=float).reshape(3, 4)
    e = N.effects_from_relative(rel, [0, 1], [2, 3])
    assert e["E_B"].tolist() == [1.5, 5.5, 9.5] and e["E_B1"].tolist() == [0.5, 4.5, 8.5]
    assert N.length_groups([5, 3, 5, 3, 7]) == [[1, 3], [0, 2], [4]]
    assert N.condition_seed("C3", "007") == 7 and N.condition_seed("C1", "007") is None


@pytest.mark.measure
@pytest.mark.skipif(not HAS_FLYVIS or not os.environ.get("FLYVIS_ROOT_DIR"),
                    reason="needs the [measure] extra and FLYVIS_ROOT_DIR with the flow/0000 ensemble")
def test_smoke_trained_model_ablation_and_patching():
    import torch

    net = N.load_trained("000", "cpu")
    classes = N.connection_classes(net)
    assert len(classes) == 604
    movie = 0.5 + 0.2 * torch.rand((30, 1, 721), generator=torch.Generator().manual_seed(0))
    r = N.response(net, movie)
    assert r.shape == (1, 30, 65) and bool(torch.isfinite(r).all())
    e = int(np.argmax([c["abs_weight"] for c in classes]))
    with N.ablated(net, e):
        r_e = N.response(net, movie)
    assert float((r_e - r).abs().max()) > 0
    assert float((N.response(net, movie) - r).abs().max()) == 0  # scale restored exactly
    v = torch.from_numpy(N.directions(2, 30, 65, b=0))
    assert N.patching_terms(net, movie, v).shape == (2, 604)
    # gradient method = patching with u = r / ||r|| as the only direction (d/d alpha = -s d/d s)
    grad = N.gradient_terms(net, movie)
    u = (r[0] / float(r.square().sum().sqrt()))[None]
    assert np.allclose(grad, N.patching_terms(net, movie, u)[0], rtol=1e-3, atol=1e-12 * grad.max())
    assert N.all_activity(net, movie).shape[0] == 30
    assert N.class_sizes(net).sum() == len(N.edge_table(net)["cls"])


# ---------------------------------------------------------------- every leaderboard method has a measurement step


def test_every_leaderboard_method_has_a_measurement_step():
    from flyvis_circuit_bench import data
    from flyvis_circuit_bench.analysis import LEADERBOARD
    from flyvis_circuit_bench.measure import run

    assert set(run.MEASURED_BY) == {*LEADERBOARD, "ceiling"}
    files = set(data.required_files())
    for method, (step, pattern, key) in run.MEASURED_BY.items():
        assert step in run.STEPS, method
        assert pattern.format(model="000") in files, method  # the file `fcb reproduce` reads
        assert key, method


def test_measure_refuses_cheap_methods_for_untrained_networks(tmp_path, capsys):
    from flyvis_circuit_bench.cli import main

    for what in ("cheap", "sae"):
        assert main(["measure", "--what", what, "--condition", "untrained", "--model", "000",
                     "--stimuli", str(tmp_path / "none.json"), "--out", str(tmp_path)]) == 2
    assert "trained networks only" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_bprime_list_is_the_study_file_and_disjoint_from_b(tmp_path):
    bp = N.load_bprime({"bodzce": [{"id": "krawedz_k000_j0_v19"}]})
    assert len(bp["bodzce"]) == 40 and bp["katy"] == [15 + 30 * k for k in range(12)]
    with pytest.raises(ValueError, match="shares"):
        N.load_bprime({"bodzce": [{"id": bp["bodzce"][0]["id"]}]})
    changed = tmp_path / "bp.json"
    changed.write_bytes(N.STIMULI_BPRIME.read_bytes().replace(b"shaman_2", b"shaman_9"))
    with pytest.raises(ValueError, match="differs"):
        N.load_bprime({"bodzce": []}, changed)


# ---------------------------------------------------------------- methods on a small synthetic network (no flyvis)


def _synthetic(seed=0, n_types=5, n_cols=12, t=30, n_stimuli=4):
    """Rate network of n_types x n_cols neurons, classes = (source type, target type) pairs, relu dynamics."""
    rng = np.random.default_rng(seed)
    typ = np.repeat(np.arange(n_types), n_cols)
    column = np.tile(np.arange(n_cols), n_types)
    n = len(typ)
    src, tgt = np.nonzero(rng.random((n, n)) < 0.15)
    pairs = sorted({(typ[s], typ[g]) for s, g in zip(src, tgt, strict=True)})
    cls = np.array([pairs.index((typ[s], typ[g])) for s, g in zip(src, tgt, strict=True)])
    w = rng.choice([-1.0, 1.0], len(pairs))[cls] * rng.integers(1, 5, len(src)) * 0.05
    acts = []
    for _ in range(n_stimuli):
        x, v = rng.random((t, n)), np.zeros(n)
        steps = []
        for k in range(t):
            v = v + 0.2 * (-v + np.bincount(tgt, weights=w * np.maximum(v[src], 0), minlength=n) + x[k])
            steps.append(v.copy())
        acts.append(np.array(steps))
    return {"typ": typ, "column": column, "src": src, "cls": cls, "w": w, "n_classes": len(pairs),
            "source_type": np.array([p[0] for p in pairs]), "acts": acts, "n_types": n_types}


def test_activity_x_weight_matches_the_definition():
    from flyvis_circuit_bench.measure import methods as MT

    s = _synthetic()
    acc = MT.ActivityAccumulator(s["w"], s["src"], s["cls"], s["n_classes"], skip=["b0"])
    for i, a in enumerate(s["acts"]):
        acc.add(f"s{i}", a)
    out = acc.result()
    fbar = np.mean([np.maximum(a, 0).mean(axis=0) for a in s["acts"]], axis=0)
    brute = [sum(abs(s["w"][j]) * fbar[s["src"][j]] for j in np.flatnonzero(s["cls"] == c))
             for c in range(s["n_classes"])]
    assert np.allclose(out["M3"], brute) and np.allclose(out["fbar"], fbar)
    with pytest.raises(ValueError, match="truth set"):
        acc.add("b0", s["acts"][0])
    with pytest.raises(ValueError, match="twice"):
        acc.add("s0", s["acts"][0])


def test_sae_gradient_matches_finite_differences():
    from flyvis_circuit_bench.measure import methods as MT

    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 6))
    p = MT.sae_init(6, 8, seed=2)
    p["b_e"] = rng.standard_normal(8) * 0.1
    g = MT.sae_gradient(p, x, 0.05)
    for k in p:
        idx = (0,) * p[k].ndim
        q = {kk: vv.copy() for kk, vv in p.items()}
        q[k][idx] += 1e-6
        r = {kk: vv.copy() for kk, vv in p.items()}
        r[k][idx] -= 1e-6
        num = (MT.sae_loss(q, x, 0.05) - MT.sae_loss(r, x, 0.05)) / 2e-6
        assert abs(num - g[k][idx]) < 1e-6, k


def test_activity_and_sae_on_synthetic_network_is_deterministic():
    from flyvis_circuit_bench.measure import methods as MT

    s = _synthetic()
    st = MT.StatisticsAccumulator()
    for i, a in enumerate(s["acts"]):
        st.add(f"s{i}", a)
    x, abs_neuron = st.result()
    assert x.shape == (len(s["typ"]), 4 * len(s["acts"]))
    abs_weight = np.bincount(s["cls"], weights=np.abs(s["w"]), minlength=s["n_classes"])
    cfg = MT.SAEConfig(dictionary_size=16, lambda_grid=(0.001, 0.1), epochs=3, minibatch=16, seed=5)
    args = (x, abs_neuron, s["typ"], s["column"], s["n_types"], abs_weight, s["source_type"], cfg)
    a, b = MT.activity_and_sae(*args), MT.activity_and_sae(*args)
    for k in ("M5", "M_akt", "g", "a_typu", "FVU_siatka", "mu", "sd"):
        assert np.array_equal(a[k], b[k]), k
    assert a["M5"].shape == a["M_akt"].shape == (s["n_classes"],)
    assert np.isfinite(a["M5"]).all() and (a["M5"] >= 0).all()
    assert float(a["lambda"]) in cfg.lambda_grid and int(a["n_walidacja"]) > 0
    # activity = sum of |weights| x mean |activity| of the source type
    a_type = [np.abs(np.concatenate(s["acts"])[:, s["typ"] == t]).mean() for t in range(s["n_types"])]
    assert np.allclose(a["M_akt"], abs_weight * np.array(a_type)[s["source_type"]])
    # validation neurons are whole columns
    train, val = MT.column_split(s["column"], cfg.seed)
    assert not set(s["column"][train]) & set(s["column"][val])
    assert MT.SAE.to_dict()["dictionary_size"] == 640 and MT.SAE.lambda_grid == (0.001, 0.003, 0.01, 0.03, 0.1)
