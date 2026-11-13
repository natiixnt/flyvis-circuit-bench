"""Connectome pictures: layout rules, and a smoke test of the figures on synthetic data."""

import json
from pathlib import Path

import numpy as np
import pytest
import studydata as study_data

from flyvis_circuit_bench import connectome as C
from flyvis_circuit_bench.data import MODELS_POOLED

ROOT = Path(__file__).resolve().parents[1]
DATA = study_data.DATA
TYPES = [t for _, ts in C.LAYERS for t in ts]
RESULTS = {"v1_1": {"z1": {"table": [{"method": "S", "k": 61, "precision_at_k": {"pooled": {"mean": 0.2836}}},
                                     {"method": "P", "k": 61, "precision_at_k": {"pooled": {"mean": 0.9}}}]}}}


def synthetic(n: int = 150, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    src = [TYPES[i] for i in rng.integers(0, len(TYPES), n)]
    dst = [TYPES[i] for i in rng.integers(0, len(TYPES), n)]
    src[0], dst[0] = "R1", "R1"  # a self connection
    src[1], dst[1] = "Mi1", "Mi4"  # inside one column
    syn = rng.lognormal(mean=8, sigma=1.5, size=n)
    base = rng.lognormal(sigma=2, size=n)
    models = {m: {"B_C1": {"E_B": base * rng.lognormal(sigma=0.3, size=n), "S": syn, "zrodlo": np.array(src),
                           "cel": np.array(dst)}} for m in MODELS_POOLED}
    return {"models": models}


def test_every_cell_type_has_exactly_one_layer_and_position():
    assert len(TYPES) == len(set(TYPES)) == 65
    pos = C.positions()
    assert set(pos) == set(TYPES) and len(set(pos.values())) == 65
    assert all(0.0 <= y <= 1.0 for _, y in pos.values())
    assert [t for t, _ in C.LAYERS][0] == "Retina" and C.LAYERS[0][1][0] == "R1"


def test_groups_follow_the_flyvis_layout_and_unplaced_types_stay_other():
    cols = dict(C.LAYERS)
    assert set(cols["Retina"]) == {f"R{i}" for i in range(1, 9)}  # layout "retina"
    intermediate = set(cols["Lamina"]) | set(cols["Medulla intrinsic"]) | set(cols["CT1"]) | set(cols["Other"])
    assert len(intermediate) == 23 and "L1" in intermediate and "CT1(M10)" in intermediate  # layout "intermediate"
    output = set(cols["T-shaped"]) | set(cols["Transmedullary"])
    assert len(output) == 34 and {"T1", "T4a", "T5d", "Tm3", "TmY15"} <= output  # layout "output"
    assert set(cols["Other"]) == {"Am", "C2", "C3"}
    assert C.KEY_LABELS <= set(TYPES) and len(C.KEY_LABELS) == 12


def test_unknown_cell_type_is_an_error_not_a_silent_drop():
    d = synthetic()
    d["models"]["000"]["B_C1"]["zrodlo"][3] = "NotACell"
    with pytest.raises(ValueError, match="without a layer"):
        C.build_table(d)


def test_table_top_and_line_width_rules():
    t = C.build_table(synthetic())
    assert t["top"].sum() == C.TOP and len(t["effect"]) == 150 and t["n_models"] == 10
    assert t["effect"][t["top"]].min() >= t["effect"][~t["top"]].max()
    w = C.line_width(np.array([1e3, 1e4, 1e5]), 1e5)
    assert (np.diff(w) > 0).all() and w[-1] == pytest.approx(4.5)


def test_edge_curves_start_and_end_at_the_nodes():
    pos = C.positions()
    for a, b in (("R1", "L1"), ("Mi1", "Mi4"), ("Mi4", "Mi1"), ("R1", "R1"), ("T5d", "R1")):
        c = C.edge_curve(pos[a], pos[b])
        assert np.allclose(c[0], pos[a]) and np.allclose(c[-1], pos[b])
    assert C.edge_curve(pos["Mi1"], pos["Mi4"])[18, 0] > pos["Mi1"][0]  # downward edge bulges right
    assert C.edge_curve(pos["Mi4"], pos["Mi1"])[18, 0] < pos["Mi1"][0]  # upward edge bulges left


def test_figures_exact_size_vector_svg_and_deterministic(tmp_path):
    import struct

    d = synthetic()
    a = C.make_figures(d, tmp_path / "a", RESULTS)
    b = C.make_figures(d, tmp_path / "b", RESULTS)
    assert sorted(p.name for p in a) == ["connectome-dark.png", "connectome-dark.svg", "connectome-light.png",
                                         "connectome-light.svg", "social-preview.png"]
    header = (tmp_path / "a" / "social-preview.png").read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", header[16:24]) == C.SOCIAL_PX
    for p in a:
        if p.suffix == ".svg":
            text = p.read_text(encoding="utf-8")
            assert "<svg" in text and "base64" not in text and "<image" not in text, p.name
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes(), pa.name


@pytest.mark.skipif(not (DATA / "SHA256SUMS").is_file(), reason="study data missing")
def test_real_package_has_604_classes_and_61_top():
    from flyvis_circuit_bench import data

    t = C.build_table(data.load(DATA))
    assert len(t["effect"]) == 604 and t["top"].sum() == C.TOP and t["n_models"] == 10
    assert {*t["source"], *t["target"]} <= set(TYPES)


def test_highlight_rule_thick_outside_top_in_nine_models_thin_inside_in_five():
    t = C.build_table(synthetic())
    t["n_top"] = np.full(150, 4, dtype=int)  # in the top 61 in 4 of 10: neither group
    t["n_top"][[3, 4, 5, 6, 7]] = [10, 5, 6, 4, 5]
    t["n_top"][[8, 9, 10, 11]] = [0, 1, 2, 0]
    sel = C.highlight_classes(t)
    s = t["synapses"]
    in5 = np.flatnonzero(t["n_top"] >= 5)
    assert sorted(sel["thin"]) == sorted(in5[np.argsort(s[in5])][:3])
    assert set(sel["thin"]) <= {3, 4, 5, 7} and len(sel["thin"]) == 3
    assert set(sel["thick"]) <= {8, 9, 11} and 10 not in sel["thick"]  # 2 of 10 inside: not 9 outside
    assert C.highlight_classes(t) == sel and not set(sel["thick"]) & set(sel["thin"])


def test_group_widths_follow_the_drawn_widths():
    t = C.build_table(synthetic())
    sel = {"thick": [0, 1], "thin": [2]}
    gw = C.group_widths(t, sel)
    smax = t["synapses"].max()
    assert gw["thin"] == pytest.approx(float(C.preview_line_width(t["synapses"][2], smax)))
    assert gw["thick"] == pytest.approx(float(C.preview_line_width(t["synapses"][[0, 1]], smax).mean()))


def test_synapse_hits_comes_from_results_and_fails_without_them():
    assert C.synapse_hits(RESULTS) == 17
    with pytest.raises(ValueError, match="v1_1.z1"):
        C.synapse_hits({})


@pytest.mark.skipif(not (DATA / "SHA256SUMS").is_file(), reason="study data missing")
@pytest.mark.skipif(not (DATA / "SHA256SUMS").is_file(), reason="study data missing")
def test_real_package_highlights_are_the_documented_classes():
    from flyvis_circuit_bench import data

    t = C.build_table(data.load(DATA))
    sel = C.highlight_classes(t)
    name = lambda i: f"{t['source'][i]}>{t['target'][i]}"  # noqa: E731
    assert [name(i) for i in sel["thick"]] == ["Tm4>TmY15", "L1>C3", "L2>Tm4"]
    assert [name(i) for i in sel["thin"]] == ["L2>R2", "L4>R5", "R8>L1"]
    gw = C.group_widths(t, sel)
    assert 2.8 <= gw["thick"] / gw["thin"] <= 4.0  # brown clearly thicker than blue
    results = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    assert C.synapse_hits(results) == 17


def _read(path):
    import matplotlib.image as mpimg

    return np.rint(mpimg.imread(path)[..., :3] * 255).astype(int)


def test_quantized_png_is_exact_for_few_colours_and_deterministic(tmp_path):
    rng = np.random.default_rng(1)
    palette = np.array([[255, 255, 255], [13, 17, 23], [0, 114, 178], [139, 107, 74], [221, 225, 230]], dtype=np.uint8)
    rgba = np.full((40, 60, 4), 255, dtype=np.uint8)
    rgba[..., :3] = palette[rng.integers(0, 5, size=(40, 60))]
    a, b = C.quantized_png(rgba), C.quantized_png(rgba)
    assert a == b and a[:8] == b"\x89PNG\r\n\x1a\n"
    (tmp_path / "q.png").write_bytes(a)
    assert (_read(tmp_path / "q.png") == rgba[..., :3]).all()


def test_quantized_png_gives_a_gradient_region_a_palette_of_its_own(tmp_path):
    h, w = 80, 300
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    rng = np.random.default_rng(2)
    rgba[:60, :, :3] = rng.integers(0, 256, size=(60, w, 3))  # a lot of different colours elsewhere
    ramp = np.linspace(0, 255, 256).astype(np.uint8)[:w] if w <= 256 else np.linspace(0, 255, w).astype(np.uint8)
    rgba[65:75, :, 0], rgba[65:75, :, 1], rgba[65:75, :, 2] = ramp, 255 - ramp, 128
    (tmp_path / "plain.png").write_bytes(C.quantized_png(rgba))
    (tmp_path / "region.png").write_bytes(C.quantized_png(rgba, regions=[(0, 65, w, 75)]))
    err = lambda f: np.abs(_read(f)[65:75] - rgba[65:75, :, :3]).max()  # noqa: E731
    assert err(tmp_path / "region.png") <= 8 < err(tmp_path / "plain.png")


@pytest.mark.skipif(not (DATA / "SHA256SUMS").is_file(), reason="study data missing")
def test_real_package_connectome_pngs_are_under_600_kb(tmp_path):
    from flyvis_circuit_bench import data

    results = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    paths = C.make_figures(data.load(DATA), tmp_path, results)
    for p in paths:
        if p.name.startswith("connectome") and p.suffix == ".png":
            assert p.stat().st_size < 600 * 1024, (p.name, p.stat().st_size)
