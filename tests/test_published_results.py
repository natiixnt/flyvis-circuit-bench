"""The committed results are byte-for-byte what `fcb reproduce` computes from the real measurement package.

Runs when the package is available: set FCB_DATA to its directory, or extract it to
`data/measurements-v1.1.0` (what `fcb download` does). Skipped otherwise.
"""

import json
import os
import re
from pathlib import Path

import pytest
import studydata as study_data

ROOT = Path(__file__).resolve().parents[1]
DATA = study_data.DATA


def test_committed_results_contain_the_headline():
    r = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    d = r["headline"]["D_C1_N5"]
    assert round(d["mean"], 3) == 0.225 and [round(x, 3) for x in d["ci_bootstrap"]] == [0.196, 0.256]
    assert r["headline"]["W1_failed"] is False
    assert {c["d10"] for c in r["replication"]["combination"].values()} == {"survives", "cheap methods survive"}


@pytest.mark.data
@pytest.mark.skipif(not DATA.is_dir(), reason="measurement package not available (set FCB_DATA or run fcb download)")
def test_reproduce_matches_committed_results(tmp_path):
    from flyvis_circuit_bench.cli import main

    rc = main(["reproduce", "--data", str(DATA), "--json", str(ROOT / "results" / "results.json"),
               "--markdown", str(ROOT / "docs" / "results.md"), "--check"])
    assert rc == 0


READMES = ["README.md", "README.pl.md"]
TLDR_FACTS = ("share_P", "auroc_P", "share_S", "auroc_S", "p61_count_S", "share_A", "auroc_A", "gap_closed_A",
              "share_M1", "auroc_M1", "auroc_M5", "dd_top61",
              "auroc_act", "share_act", "auroc_S_act", "share_S_act")
# Numbers in the README prose that are not results: versions, sizes, constants of the method, citation, years.
NOT_RESULTS = {"1.0.0", "1.0.1", "1.1.0", "38.5", "40.3", "31", "91", "121", "1000",
               "3.11", "3.12", "3.13", "0.01", "0.5", "1132", "1140", "634", "2024", "2026",
               "604", "61", "10", "32", "40", "24", "16", "000", "004", "005", "009", "5", "4", "1", "0", "100%", "0%",
               "20261101", "10%", "95%", "4.0", "440", "520"}
# Numbers quoted from other papers (docs/related_work.md gives the source and page of each).
# FlyHack, Table 3 (p. 8): first-order EAP zero to zero; truth = top 20, 20 pairs. 0.974 and 0.777: mean ablation
# (docs/v1.1/mean-ablation.md, research repository, not recomputed here).
EXTERNAL = {"0.927", "20", "0.974", "0.777"}
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)*%?(?![\w])")


def _results() -> dict:
    return json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))


def _prose(text: str, r: dict) -> str:
    """README without code blocks, the generated table, links, image sources and sha256 sums."""
    from flyvis_circuit_bench.report import leaderboard_table

    text = text.replace(leaderboard_table(r), "")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)|(?:href|src|srcset)=\"[^\"]*\"|sha256:[0-9a-f]+|https?://\S+", "", text)
    return re.sub(r"(?m)^\s*\d+\. ", "", text)  # numbered list items


@pytest.mark.parametrize("readme", READMES)
def test_readme_has_one_table_the_generated_leaderboard(readme):
    from flyvis_circuit_bench.report import leaderboard_table, main_table

    text = (ROOT / readme).read_text(encoding="utf-8")
    r = _results()
    assert leaderboard_table(r) in text
    assert main_table(r) not in text
    assert len(re.findall(r"(?m)^\|---", text)) == 1


@pytest.mark.parametrize("readme", READMES)
def test_tldr_quotes_the_computed_numbers(readme):
    from flyvis_circuit_bench.report import readme_facts

    facts = readme_facts(_results())
    tldr = (ROOT / readme).read_text(encoding="utf-8").split("\n## ")[1]
    missing = [k for k in TLDR_FACTS if facts[k] not in tldr]
    assert missing == [], missing


@pytest.mark.parametrize("readme", READMES)
def test_every_number_in_the_prose_is_a_computed_one(readme):
    """A changed or invented number anywhere in the README text fails: it must be one of `readme_facts`."""
    from flyvis_circuit_bench.report import readme_facts

    r = _results()
    allowed = set(NOT_RESULTS) | EXTERNAL
    for v in readme_facts(r).values():
        allowed |= set(NUMBER.findall(v))
    found = NUMBER.findall(_prose((ROOT / readme).read_text(encoding="utf-8"), r))
    assert len(found) > 40
    stray = sorted({x for x in found if x not in allowed})
    assert stray == [], stray


V11_FACTS = (
    "v11_z1_S_lo", "v11_z1_S_hi", "v11_z1_P_lo", "v11_z1_P_hi", "v11_z1_swaps", "v11_z2_spearman",
    "v11_z2_P_edges", "v11_z2_P_sintel", "v11_z2_S_edges", "v11_z2_S_sintel", "v11_z3_median_spearman",
    "v11_z3_core", "v11_z3_core_min", "v11_z3_split_half", "v11_z3_other_model", "v11_z3_anatomy",
    "v11_z3_better", "v11_z3_pairs", "v11_z4_spearman", "v11_z4_both", "v11_z4_high_low", "v11_z6_spearman",
    "v11_z6_masking", "v11_z6_reverse", "v11_z6_exploding", "v11_z6_pairs", "v11_z6_types")


@pytest.mark.parametrize("readme", READMES)
def test_v11_section_quotes_the_computed_numbers(readme):
    """Every number of the v1.1 section comes from `results.json` (plus the mean-ablation numbers of its page)."""
    from flyvis_circuit_bench.report import readme_facts

    facts = readme_facts(_results())
    text = (ROOT / readme).read_text(encoding="utf-8")
    start = text.index("## v1.1")
    section = text[start:text.index("\n## ", start + 5)]
    missing = [k for k in V11_FACTS if facts[k] not in section]
    assert missing == [], missing
    tldr = text.split("\n## ")[1]
    assert facts["v11_z3_median_spearman"] in tldr and "v1.1" in tldr
    page = (ROOT / "docs" / "v1.1" / "mean-ablation.md").read_text(encoding="utf-8")
    for number in ("0.974", "0.777", "0.987", "0.763"):
        assert number in section and number in page


def test_limitations_page_quotes_the_computed_v11_numbers():
    """The three v1.1 limitations of docs/limitations.md use numbers of results.json, nothing else."""
    from flyvis_circuit_bench.report import readme_facts

    facts = readme_facts(_results())
    text = (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    start = text.index("- **The effect of a class varies between models")
    block = text[start:text.index("- **Small N.**", start)]
    for key in ("v11_z3_median_spearman", "v11_z3_core", "v11_z3_core_min", "v11_z4_spearman", "v11_z4_high_low",
                "v11_z6_exploding", "v11_z6_pairs"):
        assert facts[key] in block, key
    allowed = set(NOT_RESULTS) | {x for v in facts.values() for x in NUMBER.findall(v)} | {"3", "9", "10"}
    stray = sorted({x for x in NUMBER.findall(_prose(block, _results())) if x not in allowed})
    assert stray == [], stray


def test_number_check_catches_a_changed_v11_value():
    from flyvis_circuit_bench.report import readme_facts

    r = _results()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    facts = readme_facts(r)
    allowed = set(NOT_RESULTS) | EXTERNAL | {x for v in facts.values() for x in NUMBER.findall(v)}
    changed = text.replace(facts["v11_z4_spearman"], "0.612", 1)
    assert changed != text
    assert "0.612" in {x for x in NUMBER.findall(_prose(changed, r)) if x not in allowed}


def test_number_check_catches_a_changed_value():
    from flyvis_circuit_bench.report import readme_facts

    r = _results()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    facts = readme_facts(r)
    allowed = set(NOT_RESULTS) | {x for v in facts.values() for x in NUMBER.findall(v)}
    changed = text.replace(facts["auroc_M5"], "0.883", 1)  # first occurrence: the TL;DR, before the table
    assert changed != text
    assert "0.883" in {x for x in NUMBER.findall(_prose(changed, r)) if x not in allowed}


def _code_blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("readme", READMES)
def test_add_your_method_quotes_the_computed_example_score(readme):
    from flyvis_circuit_bench.report import readme_facts

    facts = readme_facts(_results())
    bash = "".join(_code_blocks((ROOT / readme).read_text(encoding="utf-8"), "bash"))
    assert f"# scores {facts['example_auroc']} AUROC" in bash
    assert len(re.findall(r"# scores \d\.\d+ AUROC", bash)) == 1


def test_example_score_check_catches_a_changed_value():
    from flyvis_circuit_bench.report import readme_facts

    facts = readme_facts(_results())
    text = (ROOT / "README.md").read_text(encoding="utf-8").replace(f"# scores {facts['example_auroc']}",
                                                                     "# scores 0.837", 1)
    assert f"# scores {facts['example_auroc']} AUROC" not in "".join(_code_blocks(text, "bash"))


@pytest.mark.data
@pytest.mark.skipif(not DATA.is_dir(), reason="measurement package not available (set FCB_DATA or run fcb download)")
def test_readme_example_scores_what_the_readme_says(tmp_path, monkeypatch):
    """Runs the Python example of the README on the real package and scores it as `fcb score` does."""
    from flyvis_circuit_bench import score
    from flyvis_circuit_bench.report import f, readme_facts

    code = _code_blocks((ROOT / "README.md").read_text(encoding="utf-8"), "python")[0]
    monkeypatch.chdir(tmp_path)
    # the test runs the README example on the local package on purpose
    exec(code.replace('"data/measurements-v1.1.0"', repr(str(DATA))), {})  # noqa: S102  # nosec B102
    result = score.evaluate(score.load_scores(tmp_path / "my_method.npz"), DATA)
    assert f(result["AUROC"]["pooled"]["mean"]) == readme_facts(_results())["example_auroc"]


def test_study_data_is_package_1_1_0():
    """With REQUIRE_STUDY_DATA=1 the real package must be 1.1.0 (it has task/ and silence/ for Z4 and Z6)."""
    from flyvis_circuit_bench import data

    if os.environ.get("REQUIRE_STUDY_DATA") != "1":
        pytest.skip("only checked with REQUIRE_STUDY_DATA=1")
    assert data.package_version(DATA) == "1.1.0", f"{DATA}: package 1.1.0 required (fcb download)"
    assert data.verify(DATA) == []
    assert all((DATA / rel).is_file() for rel in data.v11_files())
