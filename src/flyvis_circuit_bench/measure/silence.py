"""Step `silence` (v1.1, Z6): silencing a whole cell type at once.

Silencing the type A = zeroing the synaptic scale of every connection class whose source type is A, all at once,
then the same truth metric as for one class: E_B(A) = mean over the 40 stimuli of B of
||r_A(b) - r(b)||^2 / ||r(b)||^2 (`network.ablation_effect`), with the response of the central column of every cell
type, the same initial state and integration as the single-class ablations.

The plan says 64 types; the 604 classes of the study have 63 distinct source types (65 cell types, two never occur
as a source), so 63 are measured and the number is reported as it is.

The pure driver `silence_effects` needs no flyvis and is unit tested; `measure_silence` needs the `[measure]` extra.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import network as N


def type_groups(sources) -> list[tuple[str, list[int]]]:
    """[(source type, class indices)] sorted by type name, every class in exactly one group."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(sources):
        groups.setdefault(str(s), []).append(i)
    return [(t, groups[t]) for t in sorted(groups)]


def silence_effects(respond: Callable[[], list[np.ndarray]],
                    silence: Callable[[list[int]], contextlib.AbstractContextManager],
                    groups: list[tuple[str, list[int]]], types: list[str] | None = None) -> np.ndarray:
    """Relative effect (n_types, n_stimuli) of silencing every type of `groups`; types not in `types` stay nan.

    `respond()` returns the per-stimulus responses of the network in its current state; `silence(indices)` is a
    context manager that zeroes those classes and restores them. A non-finite response gives nan.
    """
    r_ref = respond()
    rel = np.full((len(groups), len(r_ref)), np.nan)
    for g, (name, idx) in enumerate(groups):
        if types is not None and name not in types:
            continue
        with silence(idx):
            r = respond()
        if all(np.isfinite(x).all() for x in r):
            rel[g] = N.ablation_effect(r_ref, r)
    return rel


@contextlib.contextmanager
def silenced(net, indices: list[int]):
    """Scale 0 for every class in `indices`, exact previous values restored afterwards."""
    p = net.edges_syn_strength
    old = p.data[indices].clone()
    p.data[indices] = 0.0
    try:
        yield
    finally:
        p.data[indices] = old


def measure_silence(model: str, stimuli: Path, out: Path, device: str = "cpu", sintel_dir: Path | None = None,
                    types: list[str] | None = None) -> Path:
    """Silencing effect of every source type of one trained model; `types` restricts it (smoke and cost runs)."""
    from . import run

    N.limit_vram(device)
    spec, items = N.load_stimuli(stimuli, sintel_dir)
    ids = [i for i, _ in items]
    b1, b2 = [ids.index(i) for i in spec["B1"]], [ids.index(i) for i in spec["B2"]]
    t0 = time.perf_counter()
    net = N.load_trained_on_flyvis_device(model, device)
    movies = [m.to(device) for _, m in items]
    cls = N.connection_classes(net)
    groups = type_groups([c["source"] for c in cls])
    rel = silence_effects(lambda: run.responses(net, movies), lambda idx: silenced(net, idx), groups, types)
    run.torch_cpu()
    meta = {"condition": "C1", "model": model, "device": device, "n_types": len(groups),
            "types_measured": int(np.isfinite(rel[:, 0]).sum()), "seconds": time.perf_counter() - t0,
            "definition": "all classes with source type A set to scale 0 at once; E as for one class"}
    path = Path(out) / "silence" / f"C1_{model}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **N.effects_from_relative(rel, b1, b2), wzgledne=rel, tipy=np.array([g[0] for g in groups]),
             n_klas=np.array([len(g[1]) for g in groups]), bodzce=np.array(ids), meta=json.dumps(meta))
    return path
