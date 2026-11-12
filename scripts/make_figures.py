"""Render every figure (PNG + SVG, light + dark) from results/results.json into docs/figures/.

    uv run python scripts/make_figures.py [--results results/results.json] [--out docs/figures]

Deterministic: same results file, same bytes (fixed seed, no timestamps).
"""

from __future__ import annotations

import sys

from flyvis_circuit_bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["figures", *sys.argv[1:]]))
