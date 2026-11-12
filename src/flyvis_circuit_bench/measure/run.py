"""Re-measure one model: every method of the leaderboard and the ablation truth, file by file of the package.

    uv sync --extra measure
    uv run fcb measure --what all --condition trained --model 000 --stimuli data/measurements-v1.0.0/stimuli_B.json

Steps (`--what`), each writing the file of the measurement package with the same keys, so `fcb reproduce` and the
tests can use it:

    ablation   ablation/<C>_<model>.npz      truth E (604 classes), synapse count, sum of |weights|, patching;
                                             1-3 hours on a laptop CPU, about 15 minutes on a recent GPU
    cheap      cheap_methods/C1_<model>.npz  gradient (40 backward passes) and activity x weight (B'); minutes
    sae        activity_sae/C1_<model>.npz   activity and SAE features (B', SAE fit in numpy); a few minutes
    structure  structure.npz                 class size (connectome only); seconds

`cheap` and `sae` exist for trained networks only, as in the study. The second stimulus set (F2) has no
measurement step in v1.0.0 (its outputs are in the package).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import methods as MT
from . import network as N

STEPS = ("ablation", "cheap", "sae", "structure")
# Every method of the leaderboard: (step of `fcb measure`, file of the package, array key in that file).
MEASURED_BY = {
    "P": ("ablation", "ablation/C1_{model}.npz", "P"),
    "A": ("ablation", "ablation/C1_{model}.npz", "A"),
    "S": ("ablation", "ablation/C1_{model}.npz", "S"),
    "ceiling": ("ablation", "ablation/C1_{model}.npz", "E_B1"),
    "M1": ("cheap", "cheap_methods/C1_{model}.npz", "M1"),
    "M3": ("cheap", "cheap_methods/C1_{model}.npz", "M3"),
    "M_akt": ("sae", "activity_sae/C1_{model}.npz", "M_akt"),
    "M5": ("sae", "activity_sae/C1_{model}.npz", "M5"),
    "size": ("structure", "structure.npz", "n_krawedzi"),
}


def responses(net, movies, batch: bool = True) -> list[np.ndarray]:
    """r(b) for every stimulus, list of (T_b, n_types) float64 arrays."""
    import torch

    out: list = [None] * len(movies)
    with torch.no_grad():
        if batch:
            for g in N.length_groups([int(m.shape[0]) for m in movies]):
                r = N.response(net, torch.stack([movies[i] for i in g])).cpu().double().numpy()
                for j, i in enumerate(g):
                    out[i] = r[j]
        else:
            for i, m in enumerate(movies):
                out[i] = N.response(net, m)[0].cpu().double().numpy()
    return out


def network(condition: str, model: str, device: str):
    if condition == "C3":
        return N.load_initial(model, seed=N.condition_seed("C3", model), device=device)
    return N.load_trained(model, device)


def measure_pair(condition: str, model: str, stimuli: Path, out: Path, device: str = "cpu", k: int = 32,
                 sintel_dir: Path | None = None, classes: list[int] | None = None) -> Path:
    """Full measurement of one pair; `classes` restricts the ablations (smoke tests), E of others stays nan."""
    import torch

    spec, items = N.load_stimuli(stimuli, sintel_dir)
    ids = [i for i, _ in items]
    b1 = [ids.index(i) for i in spec["B1"]]
    b2 = [ids.index(i) for i in spec["B2"]]
    t0 = time.perf_counter()
    net = network(condition, model, device)
    movies = [m.to(device) for _, m in items]
    cls = N.connection_classes(net)
    r_ref = responses(net, movies)
    if not all(np.isfinite(r).all() for r in r_ref):
        raise RuntimeError("reference response is not finite: the dynamics diverge for this pair")
    todo = range(len(cls)) if classes is None else classes
    relative = np.full((len(cls), len(movies)), np.nan)
    for e in todo:
        with N.ablated(net, e):
            relative[e] = N.ablation_effect(r_ref, responses(net, movies))
    t_abl = time.perf_counter() - t0
    terms = []
    n_types = len(N.central_cells(net))
    for b, m in enumerate(movies):
        v = torch.from_numpy(N.directions(k, int(m.shape[0]), n_types, b))
        terms.append(N.patching_terms(net, m, v))
    terms = np.stack(terms)  # (stimuli, K, classes)
    torch.set_default_device("cpu")
    meta = {"condition": condition, "model": model, "seed_condition": N.condition_seed(condition, model),
            "device": device, "K": k, "seconds": {"ablation": t_abl, "total": time.perf_counter() - t0}}
    path = Path(out) / "ablation" / f"{condition}_{model}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **N.effects_from_relative(relative, b1, b2),
        wzgledne=relative,
        S=np.array([c["synapses"] for c in cls]),
        A=np.array([c["abs_weight"] for c in cls]),
        P=terms.mean(axis=(0, 1)),
        P_B1=terms[b1].mean(axis=(0, 1)),
        P_B2=terms[b2].mean(axis=(0, 1)),
        skale=np.array([c["scale"] for c in cls]),
        r_ref_norma=np.array([float(np.sum(r * r)) for r in r_ref]),
        zrodlo=np.array([c["source"] for c in cls]),
        cel=np.array([c["target"] for c in cls]),
        bodzce=np.array(ids),
        meta=json.dumps(meta),
    )
    return path


def _movies_bprime(stimuli: Path, sintel_dir: Path | None, device: str):
    spec = json.loads(Path(stimuli).read_text(encoding="utf-8"))
    bp = N.load_bprime(spec)
    return spec, bp, [(i, m.to(device)) for i, m in N.movies_of(spec, bp["bodzce"], sintel_dir, bp["katy"])]


def measure_cheap(model: str, stimuli: Path, out: Path, device: str = "cpu", sintel_dir: Path | None = None) -> Path:
    """Gradient (`M1`, on B) and activity x weight (`M3`, on B') of one trained model."""
    spec, items = N.load_stimuli(stimuli, sintel_dir)
    _, bp, movies_bp = _movies_bprime(stimuli, sintel_dir, device)
    ids = [i for i, _ in items]
    net = N.load_trained(model, device)
    et = N.edge_table(net)
    t0 = time.perf_counter()
    acc = MT.ActivityAccumulator(et["w"], et["src"], et["cls"], et["n_classes"], skip=ids)
    for ident, m in movies_bp:
        acc.add(ident, N.all_activity(net, m))
    m3 = acc.result()
    t_m3 = time.perf_counter() - t0
    t0 = time.perf_counter()
    m1 = np.stack([N.gradient_terms(net, m.to(device)) for _, m in items])
    t_m1 = time.perf_counter() - t0
    torch_cpu()
    meta = {"condition": "C1", "model": model, "device": device, "sha256_Bprime": N.SHA256_BPRIME,
            "seconds": {"M3": t_m3, "M1": t_m1}}
    path = Path(out) / MEASURED_BY["M1"][1].format(model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, M1=m1.mean(axis=0), M3=m3["M3"], M1_b=m1.astype(np.float32), fbar=m3["fbar"].astype(np.float32),
             bodzce=np.array(ids), bodzce_bp=np.array(acc.stimuli), meta=json.dumps(meta))
    return path


def measure_activity_sae(model: str, stimuli: Path, out: Path, device: str = "cpu", sintel_dir: Path | None = None,
                         cfg: MT.SAEConfig = MT.SAE) -> Path:
    """Activity (`M_akt`) and SAE features (`M5`) of one trained model, both on B'."""
    _, bp, movies_bp = _movies_bprime(stimuli, sintel_dir, device)
    net = N.load_trained(model, device)
    cls = N.connection_classes(net)
    types, typ, column = N.nodes(net)
    t0 = time.perf_counter()
    st = MT.StatisticsAccumulator()
    for ident, m in movies_bp:
        st.add(ident, N.all_activity(net, m))
    x, abs_neuron = st.result()
    t_act = time.perf_counter() - t0
    idx = {t: i for i, t in enumerate(types)}
    res = MT.activity_and_sae(x, abs_neuron, typ, column, len(types), np.array([c["abs_weight"] for c in cls]),
                              np.array([idx[c["source"]] for c in cls]), cfg)
    torch_cpu()
    meta = {"condition": "C1", "model": model, "device": device, "sha256_Bprime": N.SHA256_BPRIME,
            "sae": cfg.to_dict(), "meets_max_fvu": res.pop("meets_max_fvu"),
            "seconds": {"activity": t_act, "M5": time.perf_counter() - t0}}
    path = Path(out) / MEASURED_BY["M5"][1].format(model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, zrodlo=np.array([c["source"] for c in cls]), cel=np.array([c["target"] for c in cls]),
             bodzce_Bp=np.array(st.stimuli), **res, meta=json.dumps(meta))
    return path


def measure_structure(out: Path, model: str = "000") -> Path:
    """Class size: neuron-to-neuron edges per class; the connectome is the same for every model."""
    net = N.load_trained(model, "cpu")
    cls = N.connection_classes(net)
    path = Path(out) / MEASURED_BY["size"][1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, n_krawedzi=N.class_sizes(net), zrodlo=np.array([c["source"] for c in cls]),
             cel=np.array([c["target"] for c in cls]))
    return path


def torch_cpu() -> None:
    import torch

    torch.set_default_device("cpu")


def measure(what: str, condition: str, model: str, stimuli: Path, out: Path, device: str = "cpu", k: int = 32,
            sintel_dir: Path | None = None) -> list[Path]:
    """One step of `STEPS` or "all" (for the untrained network: ablation and structure); `cheap` and `sae` only for
    the trained network."""
    if what == "all":
        steps = STEPS if condition == "C1" else ("ablation", "structure")
    else:
        steps = (what,)
    if condition != "C1" and set(steps) & {"cheap", "sae"}:
        raise ValueError("gradient, activity x weight, activity and SAE features exist for trained networks only")
    out_paths = []
    for step in steps:
        if step == "ablation":
            out_paths.append(measure_pair(condition, model, stimuli, out, device, k, sintel_dir))
        elif step == "cheap":
            out_paths.append(measure_cheap(model, stimuli, out, device, sintel_dir))
        elif step == "sae":
            out_paths.append(measure_activity_sae(model, stimuli, out, device, sintel_dir))
        elif step == "structure":
            out_paths.append(measure_structure(out, model))
        else:
            raise ValueError(f"unknown step {step}")
    return out_paths
