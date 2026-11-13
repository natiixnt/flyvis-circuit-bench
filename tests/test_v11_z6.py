"""Z6 analysis on a synthetic data dict (the silencing measurement is not part of the v1.0.0 package)."""

import numpy as np

from flyvis_circuit_bench.data import MODELS_POOLED
from flyvis_circuit_bench.v11 import z6

TYPES = np.array([f"T{i:02d}" for i in range(12)])
N_CLASSES = 60


def synthetic(masking: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sources = np.array([TYPES[i % 12] for i in range(N_CLASSES)])
    models = {}
    for m in MODELS_POOLED:
        e = rng.lognormal(size=N_CLASSES)
        total, _ = z6.class_sums(TYPES, sources, e)
        factor = np.ones(12)
        if masking:
            factor[:3], factor[3:5] = 0.2, 3.0  # three masking types, two reverse
        sil = total * factor * rng.lognormal(sigma=0.05, size=12)
        models[m] = {"B_C1": {"E_B": e, "zrodlo": sources}, "SIL": {"tipy": TYPES, "E_B": sil}}
    return {"models": models}


def test_class_sums_and_classify_thresholds():
    s, mx = z6.class_sums(np.array(["a", "b", "z"]), np.array(["a", "a", "b"]), np.array([1.0, 3.0, 5.0]))
    assert s.tolist() == [4.0, 5.0, 0.0] and mx.tolist() == [3.0, 5.0, 0.0]
    c = z6.classify(np.array([1.9, 2.0, 6.1, 4.0, 1.0]), np.array([4.0, 4.0, 4.0, 4.0, 0.0]))
    assert c["masking"].tolist() == [True, False, False, False, False]  # 1.9 < 0.5 x 4 = 2.0, 2.0 is not below
    assert c["reverse"].tolist() == [False, False, True, False, False]  # 6.1 > 1.5 x 4 = 6.0
    assert c["valid"].tolist() == [True, True, True, True, False]


def test_available_requires_all_ten_models():
    d = synthetic()
    assert z6.available(d)
    del d["models"]["003"]["SIL"]
    assert not z6.available(d)


def test_analyze_finds_the_planted_masking_and_reverse_types():
    r = z6.analyze(synthetic(True))
    assert [x["type"] for x in r["masking_types"]] == ["T00", "T01", "T02"] or \
        {x["type"] for x in r["masking_types"]} == {"T00", "T01", "T02"}
    assert {x["type"] for x in r["reverse_types"]} == {"T03", "T04"}
    assert r["masking_types"][0]["models_masking"] == 10
    plain = z6.analyze(synthetic(False))
    assert plain["masking_types"] == [] and plain["reverse_types"] == []
    assert plain["spearman_sum"]["pooled"]["mean"] > 0.9
    assert r["n_types"] == 12


def test_render_has_dagger_sections_and_is_deterministic():
    res = {"v1_1": {"z6": z6.analyze(synthetic())}}
    md = z6.render_markdown(res)
    assert md.startswith("# Z6 †") and "## Masking types †" in md and "## Reverse types †" in md
    assert z6.render_markdown(res) == md and res["v1_1"]["z6"]["conclusion"] in md


def test_make_figures_writes_light_and_dark_without_embedded_bitmaps(tmp_path):
    import json

    res = json.loads(json.dumps({"v1_1": {"z6": z6.analyze(synthetic())}}))
    paths = z6.make_figures(res, tmp_path)
    assert sorted(p.name for p in paths) == ["v1.1-z6-silencing-dark.png", "v1.1-z6-silencing-dark.svg",
                                             "v1.1-z6-silencing-light.png", "v1.1-z6-silencing-light.svg"]
    for p in paths:
        assert p.stat().st_size > 1000
    for p in paths:
        if p.suffix == ".svg":
            text = p.read_text(encoding="utf-8")
            assert "<svg" in text and "base64" not in text and "<image" not in text
    again = z6.make_figures(res, tmp_path / "again")
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(paths, again, strict=True))


def test_notes_in_report_and_pending_docs_file_match_the_generator():
    from pathlib import Path

    md = z6.render_markdown({"v1_1": {"z6": z6.analyze(synthetic())}})
    assert z6.CLARIFICATION in md and z6.COST_ESTIMATE in md and "2026-11-12" in md and "20 s per model" in md
    doc = Path(__file__).resolve().parents[1] / "docs" / "v1.1" / "z6.md"
    assert "# Z6 †" in doc.read_text(encoding="utf-8")
    text = doc.read_text(encoding="utf-8")
    assert text == z6.render_pending() or "Silencing effect against" in text  # pending stub or the real report
