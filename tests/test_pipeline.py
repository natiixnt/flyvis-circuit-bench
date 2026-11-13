"""End-to-end on the synthetic package: analysis, CLI, determinism, verification, decisions."""

import json
import shutil

import pytest

from flyvis_circuit_bench import analysis as A
from flyvis_circuit_bench import data as D
from flyvis_circuit_bench.cli import main


def test_synthetic_package_is_complete(package):
    assert D.verify(package) == []


def test_verify_detects_missing_and_modified_files(package, tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(package, copy)
    (copy / "structure.npz").write_bytes(b"tampered")
    (copy / "f2" / "C1_009.npz").unlink()
    problems = D.verify(copy)
    assert "checksum mismatch: structure.npz" in problems
    assert "missing: f2/C1_009.npz" in problems


def test_analysis_finds_planted_effect(results):
    h = results["headline"]
    assert h["D_C1_N5"]["sign"] == "positive" and h["W1_failed"] is False
    assert h["D_C1_N10"]["n"] == 10
    comb = results["replication"]["combination"]
    assert comb["T5.1"]["d10"] == "survives"
    assert comb["T5.4"]["d5"] is None
    assert set(results["replication"]["f2_validity"]) == set(D.MODELS_POOLED)
    assert results["controls"]["K5"]["passed"] is True


def test_week1_rows_cover_both_conditions(results):
    keys = {r["key"] for r in results["week1"]["methods"]}
    assert {"C1|S|AUROC", "C3|P|Spearman", "C1|ceiling|AUPRC"} <= keys
    labels = [r["label"] for r in results["week1"]["contrasts"]]
    assert labels[0] == "D_C1" and "D_C1 - D_C3 (AUROC, P - S)" in labels


def test_combination_rule():
    assert A.combine("survives", "survives", "survives") == "confirmed"
    assert A.combine("survives", "inconclusive", "inconclusive") == "weakened"
    assert A.combine("survives", "survives", "refuted") == "disputed (disagrees with replication)"
    assert A.combine(None, "survives", "survives").startswith("no discovery decision")
    assert A.combine("survives", "undetermined", None).startswith("undetermined")


def test_decision_t51_requires_room_below_ceiling():
    q = {"undetermined": False, "sign": "positive"}
    assert A.decide_t51(q, 0.2, True) == "survives"
    assert A.decide_t51(q, 0.05, True) == "inconclusive"  # exactly 0.05 counts as failure
    assert A.decide_t51(q, 0.2, False) == "inconclusive"
    assert A.decide_t51({"undetermined": True, "sign": "undetermined"}, 0.2, True) == "undetermined"


def test_reproduce_cli_is_deterministic_and_check_works(package, tmp_path, capsys):
    j, md = tmp_path / "r.json", tmp_path / "r.md"
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md)]) == 0
    first = (j.read_bytes(), md.read_bytes())
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md), "--check"]) == 0
    assert (j.read_bytes(), md.read_bytes()) == first
    md.write_text("edited", encoding="utf-8")
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md), "--check"]) == 1
    r = json.loads(first[0])
    assert r["headline"]["D_C1_N5"]["n"] == 5
    out = capsys.readouterr().out
    assert "| method | trained, all 10 | discovery (000-004) |" in out


def test_reproduce_refuses_incomplete_package(package, tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(package, copy)
    (copy / "ablation" / "C1_000.npz").unlink()
    with pytest.raises(SystemExit, match="incomplete or modified"):
        main(["reproduce", "--data", str(copy), "--json", str(tmp_path / "a"), "--markdown", str(tmp_path / "b")])


def test_class_order_mismatch_is_an_error(package, tmp_path):
    import numpy as np

    copy = tmp_path / "pkg"
    shutil.copytree(package, copy)
    s = dict(np.load(copy / "structure.npz"))
    s["zrodlo"] = s["zrodlo"][::-1]
    np.savez(copy / "structure.npz", **s)
    with pytest.raises(ValueError, match="class order"):
        D.load(copy)


def test_markdown_has_no_long_dashes(results):
    from flyvis_circuit_bench.report import render_markdown

    text = render_markdown(results)
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_reproduce_on_package_without_task_and_silence_keeps_their_results(package, tmp_path, capsys):
    j, md = tmp_path / "r.json", tmp_path / "r.md"
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md)]) == 0
    committed = json.loads(j.read_text(encoding="utf-8"))
    committed["v1_1"]["z4"] = {"kept": 1}  # results of a package 1.1.0 that this package cannot recompute
    committed["v1_1"]["z6"] = {"kept": 2}
    j.write_text(json.dumps(committed), encoding="utf-8")
    capsys.readouterr()
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md), "--check"]) == 0
    err = capsys.readouterr().err
    assert "not computed: Z4, Z6" in err and "package 1.1.0" in err
    committed["headline"]["D_C1_N5"]["mean"] += 0.5  # a changed v1.0 key is still red
    j.write_text(json.dumps(committed), encoding="utf-8")
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md), "--check"]) == 1
    assert main(["reproduce", "--data", str(package), "--json", str(j), "--markdown", str(md)]) == 0
    kept = json.loads(j.read_text(encoding="utf-8"))["v1_1"]
    assert kept["z4"] == {"kept": 1} and kept["z6"] == {"kept": 2} and list(kept)[-2:] == ["z4", "z6"]


def test_verify_checks_the_v1_1_files_listed_in_the_sums(package, tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(package, copy)
    (copy / "task").mkdir()
    (copy / "task" / "C1_000.npz").write_bytes(b"x")
    sums = (copy / "SHA256SUMS").read_text(encoding="utf-8")
    import hashlib

    (copy / "SHA256SUMS").write_text(sums + f"{hashlib.sha256(b'x').hexdigest()}  task/C1_000.npz\n"
                                     f"{hashlib.sha256(b'y').hexdigest()}  silence/C1_000.npz\n", encoding="utf-8")
    problems = D.verify(copy)
    assert problems == ["missing: silence/C1_000.npz"]
    (copy / "task" / "C1_000.npz").write_bytes(b"tampered")
    assert "checksum mismatch: task/C1_000.npz" in D.verify(copy)
    assert D.v11_files()[0] == "silence/C1_000.npz" and len(D.v11_files()) == 20
