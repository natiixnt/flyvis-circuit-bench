"""Analyses added after the study (v1.1, exploratory, marked with a dagger). One module per analysis.

Every module `zN.py` exposes

    KEY                                the key under `results["v1_1"]` and the name of `docs/v1.1/<KEY>.md`
    analyze(loaded) -> dict            the block of results.json (`loaded` = `data.load(...)`)
    render_markdown(results) -> str    the text of `docs/v1.1/<KEY>.md`
    make_figures(results, out) -> list of the figure files written (`results` as read back from results.json)
    available(loaded) -> bool          optional: False when the analysis needs data the package does not have

A new analysis is one new module and one name in `MODULES`; `fcb reproduce` and `fcb figures` need no edit.
"""

from __future__ import annotations

import importlib
from pathlib import Path

# The order is the order of the keys in results.json, which is published, so it is fixed (z2 was added first).
MODULES = ("z2", "z1", "z3", "z4", "z6")


def modules() -> list:
    return [importlib.import_module(f"{__name__}.{name}") for name in MODULES]


def analyze_all(loaded: dict) -> dict:
    """The `v1_1` block of results.json: every analysis whose data is present."""
    return {m.KEY: m.analyze(loaded) for m in modules() if getattr(m, "available", lambda _: True)(loaded)}


def unavailable(loaded: dict) -> list[str]:
    """Keys of the analyses whose data the package does not have (v1.0.0 has no task/ and silence/)."""
    return [m.KEY for m in modules() if not getattr(m, "available", lambda _: True)(loaded)]


def outputs(results: dict, docs_dir: Path) -> dict[Path, str]:
    """{docs/v1.1/<KEY>.md: text} for every analysis present in `results`."""
    return {Path(docs_dir) / "v1.1" / f"{m.KEY}.md": m.render_markdown(results)
            for m in modules() if m.KEY in results.get("v1_1", {})}


def make_figures(results: dict, out: Path) -> list[Path]:
    paths: list[Path] = []
    for m in modules():
        if m.KEY in results.get("v1_1", {}):
            paths += m.make_figures(results, out)
    return paths
