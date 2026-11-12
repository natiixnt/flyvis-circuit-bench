"""Repository hygiene: self-contained imports, light default path, documentation conventions."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sorted(p for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py"))
HEAVY = {"torch", "flyvis"}
# Everything the code may import: standard library, this package, declared dependencies (and pytest in tests).
# An allowlist instead of a denylist: nothing outside it, e.g. code of the original study, can sneak in.
ALLOWED = set(sys.stdlib_module_names) | {"flyvis_circuit_bench", "numpy", "scipy", "matplotlib", "pytest",
                                          "dotenv"} | HEAVY


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split(".")[0], node.col_offset) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], node.col_offset))
    return out


def test_imports_only_stdlib_package_and_declared_dependencies():
    bad = [f"{p.relative_to(ROOT)}: {m}" for p in PY for m, _ in _imports(p) if m not in ALLOWED]
    assert bad == []


def test_torch_and_flyvis_only_lazily_inside_measure():
    for p in PY:
        for m, col in _imports(p):
            if m in HEAVY:
                assert "measure" in p.parts or p.name == "test_measure.py", p
                assert col > 0, f"{p}: top-level import of {m}"


def test_texts_have_no_long_dashes():
    exts = {".md", ".py", ".toml", ".yml", ".yaml", ".cff", ".txt", ""}
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in exts
             and not any(x in p.parts for x in (".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"))]
    bad = [str(p.relative_to(ROOT)) for p in files if chr(0x2014) in p.read_text("utf-8", errors="ignore")
           or chr(0x2013) in p.read_text("utf-8", errors="ignore")]
    assert bad == []


def test_docs_links_point_to_existing_files():
    import re

    for md in [ROOT / "README.md", ROOT / "README.pl.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = md.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
        targets += re.findall(r'(?:href|src|srcset)="([^"#]+)', text)
        for target in targets:
            # ../../actions/... is the GitHub-relative URL of the CI badge, valid only on github.com
            if target.startswith(("http://", "https://", "mailto:", "../../actions/")):
                continue
            assert (md.parent / target).exists(), f"{md.name}: broken link {target}"


# Short codes of the study (methods, networks, decisions, stages); people read names, see docs/methodology.md.
JARGON = re.compile(
    r"M_akt|\bM[1-5]\b|(?<![\w/`])C[1-3](?![\w])|\bT5\.|\bE5\.|\bF2\b|\bW1\b|\b[Ww]eek \d|<" "owner>"
    r"|preregistered-yes|\bD_C|\bD\^|\bG\^|Delta S|\bK[1-5]\b|\b(?:AUROC|AUPRC|m)\([SAP]\)|\b[SP] (?:-|vs|at) ")
GLOSSARY = "\n## Glossary"
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "src", "tests", "results", "data",
             "figures"}


def _text_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".md", ".cff", ".yml", ".ipynb", ""}
                  and not SKIP_DIRS & set(p.relative_to(ROOT).parts) and p.name not in {"LICENSE", "LICENSE-data"})


def test_texts_use_names_not_study_codes():
    """Codes may appear only in the glossary table of docs/methodology.md (and in code, data and JSON keys)."""
    bad = []
    for p in _text_files():
        text = p.read_text(encoding="utf-8")
        if p.name == "methodology.md":
            text = text.split(GLOSSARY)[0]
        bad += [f"{p.relative_to(ROOT)}: {m.group()!r}" for m in JARGON.finditer(text)]
    assert len(_text_files()) >= 15
    assert bad == []


def test_jargon_check_finds_codes():
    for s in ("M_akt", "in C1 and", "(C3)", "T5.1", "week 5", "<" "owner>", "preregistered-yes", "AUROC(S)",
              "P at ceiling"):
        assert JARGON.search(s), s
    for s in ("ablation/C1_000.npz", "`C1_`", "M1_F2", "E_F2", "A method", "class size"):
        assert not JARGON.search(s), s


def test_notebook_is_clean_and_linked():
    import json

    nb = json.loads((ROOT / "notebooks" / "score_your_method.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in nb["cells"] if c["cell_type"] == "code")
    for i, src in enumerate(code):
        compile(src, f"cell {i}", "exec")
    assert any("score.evaluate" in c for c in code) and any("download.download(\"natiixnt\"" in c for c in code)
    link = "colab.research.google.com/github/natiixnt/flyvis-circuit-bench/blob/main/notebooks/score_your_method.ipynb"
    for readme in ("README.md", "README.pl.md"):
        assert link in (ROOT / readme).read_text(encoding="utf-8")


def _all_text_files() -> list[Path]:
    """Every text file of the repository, code and tests included."""
    exts = {".md", ".py", ".toml", ".yml", ".yaml", ".cff", ".txt", ".ipynb", ""}
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "data"}
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix in exts and not skip & set(p.parts))


REFUTED_PL = "Obalone: trening zmniejsza przewagę patchingu nad anatomią, a nie ją zwiększa."
OLD_WORDINGS = ("patching loses less than anatomy", "training does not widen", "nie poszerza",
                "traci mniej niż anatomia", "The mechanism hypothesis is refuted")


def test_refuted_hypothesis_has_one_wording():
    from flyvis_circuit_bench.names import REFUTED_SENTENCE

    this = Path(__file__).name
    for p in _all_text_files():
        text = p.read_text(encoding="utf-8")
        if p.name != this:
            assert not [w for w in OLD_WORDINGS if w in text], p
        # "does not widen" only inside the one sentence (names.py holds it, this test quotes it)
        if p.name not in {this, "names.py"}:
            assert text.count("does not widen") == text.count(REFUTED_SENTENCE[:-1]), p
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(REFUTED_SENTENCE) == 2  # TL;DR and negative results
    assert (ROOT / "README.pl.md").read_text(encoding="utf-8").count(REFUTED_PL) == 2
    assert REFUTED_SENTENCE[:-1] in (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")


INIT_EN = ("flyvis initializes every class strength from the connectome (weight sum = 0.01 x number of edges), so "
           "before training each weight is a fixed function of the anatomy")
INIT_PL = ("flyvis ustawia na starcie siłę każdej klasy z konektomu (suma wag = 0.01 x liczba krawędzi), więc przed "
           "treningiem każda waga jest stałą funkcją anatomii")
AUROC_EN = ("AUROC measures how well a method's ranking of the 604 classes separates the top 10% by ablation effect "
            "(61 classes).")
AUROC_PL = "AUROC mierzy, jak dobrze ranking 604 klas według metody oddziela górne 10% według efektu ablacji (61 klas)."
FLYHACK_EN = ("(FlyHack: one decoded optical-flow number per behaviour, truth = top 20, averaged over 20 model x "
              "behaviour pairs; here: central column of every cell type, truth = top 61)")
FLYHACK_PL = ("(FlyHack: jedna zdekodowana liczba przepływu optycznego na zachowanie, prawda = górne 20, średnia po 20 "
              "parach model x zachowanie; tutaj: centralna kolumna każdego typu komórki, prawda = górne 61)")
OLD_DEFINITIONS = ("AUROC ranks", "AUROC szereguje", "from the synapse counts", "initialized from synapse counts",
                   "z liczby synaps, więc")


def _flat(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_initialization_auroc_and_flyhack_have_one_wording():
    """One sentence each for the flyvis initialization, the AUROC metric and the FlyHack comparison; the frozen
    preregistration keeps its own words."""
    this = Path(__file__).name
    for p in _all_text_files():
        if p.name not in {this, "preregistration.md"}:
            assert not [w for w in OLD_DEFINITIONS if w in _flat(p)], p
    readme, readme_pl = _flat(ROOT / "README.md"), _flat(ROOT / "README.pl.md")
    for text, words in ((readme, (INIT_EN, AUROC_EN, FLYHACK_EN)), (readme_pl, (INIT_PL, AUROC_PL, FLYHACK_PL))):
        assert all(text.count(w) == 1 for w in words)
    assert INIT_EN in _flat(ROOT / "docs" / "limitations.md")
    assert AUROC_EN in _flat(ROOT / "docs" / "methodology.md")
    assert FLYHACK_EN in _flat(ROOT / "docs" / "related_work.md")


def test_package_name_has_no_study_code():
    from flyvis_circuit_bench.download import PACKAGE

    assert PACKAGE == "measurements-v1.0.0"
    bad = [str(p.relative_to(ROOT)) for p in _all_text_files()
           if re.search(r"\bw1[-_ ]?measurements", p.read_text(encoding="utf-8"), re.I)]
    assert bad == []


def test_no_priority_claims_and_one_sae():
    pattern = re.compile(r"\bfirst (?:benchmark|to)\b|\bSAEs\b", re.I)
    this = Path(__file__).name
    bad = [f"{p.relative_to(ROOT)}: {m.group()}" for p in _all_text_files() if p.name != this
           for m in pattern.finditer(p.read_text(encoding="utf-8"))]
    assert bad == []
