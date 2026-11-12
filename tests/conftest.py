"""Synthetic measurement package with the real layout (small, generated on the fly, fixed seed)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flyvis_circuit_bench.data import MODELS_POOLED, required_files

N_CLASSES = 120
N_STIMULI = 40
N_BARS = 24
TYPES = [f"R{i}" for i in range(1, 9)] + [f"L{i}" for i in range(1, 6)] + ["T4a", "T4b", "T5a", "Mi1", "Tm3"]


def _classes(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pairs = set()
    while len(pairs) < N_CLASSES:
        a, b = rng.choice(len(TYPES), size=2, replace=False)
        pairs.add((TYPES[a], TYPES[b]))
    pairs = sorted(pairs)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _noisy(rng, x, sd):
    return x * np.exp(rng.normal(0, sd, size=x.shape))


def build_package(root: Path, seed: int = 7, p_noise: float = 0.15) -> Path:
    """Write a complete synthetic package to `root` (methods: P close to E, S far from E) and SHA256SUMS."""
    rng = np.random.default_rng(seed)
    src, tgt = _classes(rng)
    n_edges = rng.integers(1, 400, size=N_CLASSES).astype(np.float64)
    stimuli = [f"stim_{i:02d}" for i in range(N_STIMULI)]
    for sub in ("ablation", "cheap_methods", "activity_sae", "f2"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    first_e = None
    for m in MODELS_POOLED:
        for c in ("C1", "C3"):
            rel = rng.lognormal(-6, 1.5, size=(N_CLASSES, 1)) * rng.lognormal(0, 0.2, size=(N_CLASSES, N_STIMULI))
            e = rel.mean(axis=1)
            half = N_STIMULI // 2
            d = {
                "E_B": e, "E_B1": rel[:, :half].mean(axis=1), "E_B2": rel[:, half:].mean(axis=1), "wzgledne": rel,
                "S": _noisy(rng, e, 1.2 if c == "C1" else 0.8), "A": _noisy(rng, e, 0.7) if c == "C1"
                else 0.01 * n_edges, "P": _noisy(rng, e, p_noise),
                "r_ref_norma": rng.uniform(1e3, 1e5, size=N_STIMULI), "zrodlo": src, "cel": tgt,
                "bodzce": np.array(stimuli), "meta": json.dumps({"condition": c, "model": m}),
            }
            np.savez(root / "ablation" / f"{c}_{m}.npz", **d)
            if c == "C1":
                if first_e is None:
                    first_e = e
                np.savez(root / "cheap_methods" / f"C1_{m}.npz", M1=_noisy(rng, e, 0.3), M3=_noisy(rng, e, 0.4),
                         meta=json.dumps({}))
                np.savez(root / "activity_sae" / f"C1_{m}.npz", M_akt=_noisy(rng, e, 0.45),
                         M5=_noisy(rng, e, 0.6), zrodlo=src, cel=tgt, meta=json.dumps({}),
                         a_typu=0.5 + 1.5 * ((np.arange(65) * 37) % 65) / 65)
                bars = rng.lognormal(-6, 1.5, size=(N_CLASSES, 1)) * rng.lognormal(0, 0.2, size=(N_CLASSES, N_BARS))
                e2 = bars.mean(axis=1)
                np.savez(root / "f2" / f"C1_{m}.npz", E_F2=e2, E_F2a=bars[:, :12].mean(axis=1),
                         E_F2b=bars[:, 12:].mean(axis=1), r_ref_norma_F2=rng.uniform(1e3, 1e5, size=N_BARS),
                         S=_noisy(rng, e2, 1.2), A=_noisy(rng, e2, 0.7), P_F2=_noisy(rng, e2, p_noise),
                         M1_F2=_noisy(rng, e2, 0.3), zrodlo=src, cel=tgt, meta=json.dumps({}))
    np.savez(root / "structure.npz", n_krawedzi=n_edges, zrodlo=src, cel=tgt)
    classes = sorted(int(i) for i in rng.choice(N_CLASSES, size=6, replace=False))
    k5 = {"klasy": classes, "E_zapisane": first_e[classes].tolist(), "E_druga_sciezka": first_e[classes].tolist()}
    (root / "k5_independent_ablation.json").write_text(json.dumps(k5), encoding="utf-8")
    (root / "stimuli_B.json").write_text(json.dumps({"bodzce": [{"id": s} for s in stimuli],
                                                     "B1": stimuli[:20], "B2": stimuli[20:]}), encoding="utf-8")
    write_sums(root)
    return root


def write_sums(root: Path) -> None:
    lines = [f"{hashlib.sha256((root / r).read_bytes()).hexdigest()}  {r}" for r in required_files()]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def package(tmp_path_factory) -> Path:
    return build_package(tmp_path_factory.mktemp("pkg") / "measurements-v1.0.0")


@pytest.fixture(scope="session")
def results(package) -> dict:
    from flyvis_circuit_bench import analysis, data

    return analysis.analyze(data.load(package))
