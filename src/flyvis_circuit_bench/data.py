"""Layout and loading of the measurement package `measurements-v1.0.0`.

    ablation/<C>_<model>.npz        ground truth E (ablation of 604 classes) + methods S, A, P; C1 and C3
    cheap_methods/C1_<model>.npz    M1 (one backward pass) and M3 (weight x source activity); C1
    activity_sae/C1_<model>.npz     M_akt (weight x mean |activity|) and M5 (sparse autoencoder); C1
    f2/C1_<model>.npz               second stimulus family F2 (moving bars): E_F2, S, A, P_F2, M1_F2; C1
    structure.npz                   neuron-to-neuron edge count per class (class-size baseline)
    k5_independent_ablation.json    control K5: 30 classes re-ablated by an independent code path
    stimuli_B.json                  frozen stimulus set B (40 stimuli) and its split into halves B1/B2
    SHA256SUMS                      sha256 of every file above

Models 000-004 were measured on Apple silicon (CPU and MPS), models 005-009 on an NVIDIA GPU (CUDA). Array
keys inside the files keep the names used during the study (some are Polish); `docs/data.md` lists them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("C1", "C3")
MODELS_ORIGINAL = ("000", "001", "002", "003", "004")
MODELS_REPLICATION = ("005", "006", "007", "008", "009")
MODELS_POOLED = MODELS_ORIGINAL + MODELS_REPLICATION
CHECKSUMS = "SHA256SUMS"


def required_files() -> list[str]:
    """Relative paths of every file `fcb reproduce` reads (sorted)."""
    out = [f"ablation/{c}_{m}.npz" for c in CONDITIONS for m in MODELS_POOLED]
    out += [f"cheap_methods/C1_{m}.npz" for m in MODELS_POOLED]
    out += [f"activity_sae/C1_{m}.npz" for m in MODELS_POOLED]
    out += [f"f2/C1_{m}.npz" for m in MODELS_POOLED]
    out += ["structure.npz", "k5_independent_ablation.json", "stimuli_B.json"]
    return sorted(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums(text: str) -> dict[str, str]:
    """Parse `sha256  relative/path` lines (the format of `sha256sum`)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        out[name.lstrip("*")] = digest.lower()
    return out


def verify(data_dir: Path) -> list[str]:
    """Problems found in `data_dir` (missing files, checksum mismatches); empty list means OK."""
    data_dir = Path(data_dir)
    problems = [f"missing: {rel}" for rel in required_files() if not (data_dir / rel).is_file()]
    sums_path = data_dir / CHECKSUMS
    if not sums_path.is_file():
        return problems + [f"missing: {CHECKSUMS}"]
    sums = read_checksums(sums_path.read_text(encoding="utf-8"))
    for rel in required_files():
        p = data_dir / rel
        if not p.is_file():
            continue
        if rel not in sums:
            problems.append(f"not listed in {CHECKSUMS}: {rel}")
        elif sha256_file(p) != sums[rel]:
            problems.append(f"checksum mismatch: {rel}")
    return problems


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if "meta" in d:
        d["meta"] = json.loads(str(d["meta"]))
    return d


def load(data_dir: Path) -> dict:
    """{"models": {model: {kind: arrays}}, "structure": arrays, "k5": dict} for the files present.

    kinds: "B_C1", "B_C3" (ablation), "M13" (cheap_methods), "M5" (activity_sae), "F2".
    """
    data_dir = Path(data_dir)
    kinds = {
        "B_C1": "ablation/C1_{m}.npz",
        "B_C3": "ablation/C3_{m}.npz",
        "M13": "cheap_methods/C1_{m}.npz",
        "M5": "activity_sae/C1_{m}.npz",
        "F2": "f2/C1_{m}.npz",
    }
    models: dict[str, dict] = {}
    for m in MODELS_POOLED:
        d = {}
        for kind, pattern in kinds.items():
            p = data_dir / pattern.format(m=m)
            if p.is_file():
                d[kind] = load_npz(p)
        models[m] = d
    structure_path = data_dir / "structure.npz"
    k5_path = data_dir / "k5_independent_ablation.json"
    out = {
        "models": models,
        "structure": load_npz(structure_path) if structure_path.is_file() else None,
        "k5": json.loads(k5_path.read_text(encoding="utf-8")) if k5_path.is_file() else None,
    }
    check_class_order(out)
    return out


def check_class_order(data: dict) -> None:
    """Every file must list the 604 classes in the same (source, target) order."""
    ref = None
    for m, kinds in sorted(data["models"].items()):
        for kind, d in sorted(kinds.items()):
            if "zrodlo" not in d:
                continue
            pairs = list(zip(d["zrodlo"].tolist(), d["cel"].tolist(), strict=True))
            if ref is None:
                ref = pairs
            elif pairs != ref:
                raise ValueError(f"class order differs in {kind} of model {m}")
    s = data.get("structure")
    if s is not None and ref is not None:
        if list(zip(s["zrodlo"].tolist(), s["cel"].tolist(), strict=True)) != ref:
            raise ValueError("class order differs in structure.npz")
