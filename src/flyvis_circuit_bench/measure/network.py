"""Access to flyvis networks: conditions C1 and C3, connection classes, ablation, responses, patching.

Needs the optional extra: `uv sync --extra measure` (flyvis 1.2.0 + torch). torch and flyvis are imported lazily,
so the pure helpers at the top of this module work without them and are unit tested.

flyvis data (ensemble `flow/0000` and, for the Sintel stimuli, the MPI Sintel training images) lives under
`FLYVIS_ROOT_DIR` (environment variable; default `./flyvis_data`). See `docs/methodology.md`, "Re-measuring".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SEED = 20261101
ENSEMBLE = "flow/0000"
DT = 1 / 50  # integration step used to train the ensemble (MultiTaskSintel dt = 0.02)

# ---------------------------------------------------------------- pure helpers (no flyvis)


def ablation_effect(r_ref: list[np.ndarray], r_abl: list[np.ndarray]) -> np.ndarray:
    """||r_e(b) - r(b)||^2 / ||r(b)||^2 for every stimulus b; E(e) is the mean of this array."""
    if len(r_ref) != len(r_abl):
        raise ValueError("different number of stimuli")
    out = []
    for a, b in zip(r_ref, r_abl, strict=True):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"different shapes {a.shape} and {b.shape}")
        denom = float(np.sum(a * a))
        if denom == 0.0:
            raise ValueError("zero reference response")
        out.append(float(np.sum((b - a) ** 2)) / denom)
    return np.array(out)


def directions(k: int, t: int, n_types: int, b: int, seed: int = SEED) -> np.ndarray:
    """Random output directions v_k ~ N(0, I) for stimulus b; the first k rows do not depend on k."""
    rng = np.random.default_rng([seed, b])
    return rng.standard_normal((k, t, n_types)).astype(np.float32) if k else np.zeros((0, t, n_types))


def effects_from_relative(relative: np.ndarray, b1: list[int], b2: list[int]) -> dict[str, np.ndarray]:
    """E on B, B1 and B2 as means of the relative effects (n_classes, n_stimuli) over stimuli."""
    return {"E_B": relative.mean(axis=1), "E_B1": relative[:, b1].mean(axis=1), "E_B2": relative[:, b2].mean(axis=1)}


def condition_seed(condition: str, model: str) -> int | None:
    """C3 uses seed = int(model) for the resting-potential initialization; C1 has no seed."""
    return int(model) if condition == "C3" else None


def length_groups(lengths: list[int]) -> list[list[int]]:
    """Stimulus indices grouped by number of frames (one batch per group), stable order."""
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(lengths):
        groups.setdefault(int(t), []).append(i)
    return [groups[t] for t in sorted(groups)]


# ---------------------------------------------------------------- flyvis (lazy imports)


def prepare_flyvis(device: str | None = None):
    """Import flyvis with its data directory; disables flyvis' automatic loading of `.env` files from parent
    directories, so no unrelated environment variables enter the process."""
    os.environ.setdefault("FLYVIS_ROOT_DIR", str(Path("flyvis_data").resolve()))
    import dotenv

    dotenv.load_dotenv = lambda *a, **k: False
    dotenv.find_dotenv = lambda *a, **k: ""
    import flyvis
    import torch

    if device is not None:
        torch.set_default_device(device)
    return flyvis


@contextmanager
def default_device(device: str):
    """flyvis creates some tensors on the default torch device, so we switch it for the block."""
    prepare_flyvis()
    import torch

    old = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(old)


def _no_grad(net, device: str = "cpu"):
    """Eval mode without gradients; the network is built on CPU (float64 init) and then moved with the index
    tensors flyvis keeps outside module buffers."""
    import torch

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if device != "cpu":
        net = net.to(device)
        net._source_indices = net._source_indices.to(device)
        net._target_indices = net._target_indices.to(device)
        for group in (net.node_params, net.edge_params):
            for par in group.values():
                par.indices = par.indices.to(device)
                par.readers = {k: v.to(device) for k, v in par.readers.items()}
    torch.set_default_device(device)
    return net


def load_trained(model: str, device: str = "cpu"):
    """C1: trained model of the ensemble (checkpoint `best`)."""
    prepare_flyvis("cpu")
    import flyvis
    from flyvis.network import NetworkView

    net = NetworkView(flyvis.results_dir / ENSEMBLE / model).init_network(checkpoint="best")
    return _no_grad(net, device)


def load_initial(model: str, seed: int | None = None, device: str = "cpu"):
    """C3: network from the flyvis initialization of the model's config (no checkpoint).

    The only random parameter of the ensemble config is the resting potential (bias); `seed` replaces its seed.
    Synaptic scales are 0.01 / mean synapse count per class, time constants 0.05.
    """
    prepare_flyvis("cpu")
    import flyvis
    from flyvis.network import Network, NetworkView

    cfg = NetworkView(flyvis.results_dir / ENSEMBLE / model).dir.config.network.to_dict()
    if seed is not None:
        cfg["node_config"]["bias"]["seed"] = int(seed)
    return _no_grad(Network(**cfg), device)


def scales(net) -> np.ndarray:
    return net.edges_syn_strength.detach().cpu().numpy().copy()


def connection_classes(net) -> list[dict]:
    """Type-to-type classes: index, source, target, sign, total synapse count (S), total |weight| (A), scale."""
    p = net.edge_params.syn_strength
    idx = p.indices.detach().cpu().numpy()
    n = len(p.keys)
    e = net.connectome.edges
    n_syn = np.asarray(e.n_syn[:], dtype=np.float64)
    syn_model = (net.edge_params.syn_count.semantic_values.detach().cpu().numpy()[
        net.edge_params.syn_count.indices.detach().cpu().numpy()]).astype(np.float64)
    sign_edge = np.asarray(e.sign[:], dtype=np.float64)
    s = scales(net).astype(np.float64)
    synapses = np.bincount(idx, weights=n_syn, minlength=n)
    weight = np.bincount(idx, weights=np.abs(sign_edge * syn_model * s[idx]), minlength=n)
    sign = net.edge_params.sign.semantic_values.detach().cpu().numpy()
    if len(sign) != n:
        raise ValueError("number of signs differs from number of classes")
    return [{"index": i, "source": str(p.keys[i][0]), "target": str(p.keys[i][1]), "sign": int(sign[i]),
             "synapses": float(synapses[i]), "scale": float(s[i]), "abs_weight": float(weight[i])} for i in range(n)]


@contextmanager
def ablated(net, e: int):
    """Ablation of class e: scale 0 inside the block, exact previous value restored afterwards."""
    p = net.edges_syn_strength
    old = p.data[e].clone()
    p.data[e] = 0.0
    try:
        yield
    finally:
        p.data[e] = old


def central_cells(net) -> np.ndarray:
    """Node indices of the central column, one per cell type."""
    return np.asarray(net.connectome.central_cells_index[:])


def _repeat_state(net, state, b: int):
    from flyvis.utils.tensor_utils import AutoDeref

    def rep(d):
        return AutoDeref(**{k: v.repeat(b, *([1] * (v.ndim - 1))) for k, v in d.items()})

    return net._state_api(AutoDeref(nodes=rep(state.nodes), edges=rep(state.get("edges", {}))))


def response(net, movie, dt: float = DT, grad: bool = False):
    """r(b): activity of the central column of every cell type, tensor (B, T, n_types).

    The initial state is 1 s of grey input, as in `Network.simulate`; it is the same for every stimulus, so it is
    computed once with batch 1 and repeated.
    """
    import torch

    if movie.ndim == 3:
        movie = movie[None]
    b, t = movie.shape[:2]
    idx = torch.as_tensor(central_cells(net), device=movie.device)
    with torch.set_grad_enabled(grad):
        state = net.steady_state(1.0, dt, 1, grad=grad)
        if b > 1:
            state = _repeat_state(net, state, b)
        net.stimulus.zero(b, t)
        net.stimulus.add_input(movie)
        act = net.forward(net.stimulus(), dt, state)
    return act[:, :, idx]


def patching_terms(net, movie, v, dt: float = DT) -> np.ndarray:
    """P terms for one stimulus, array (K, n_classes): (s_e * d(v_k . r)/d s_e)^2 / ||r||^2."""
    import torch

    p = net.edges_syn_strength
    p.requires_grad_(True)
    try:
        r = response(net, movie, dt, grad=True)[0]
        v = v.to(r.device)
        norm = float((r.detach() ** 2).sum())
        out = []
        for k in range(v.shape[0]):
            (g,) = torch.autograd.grad((v[k] * r).sum(), p, retain_graph=k < v.shape[0] - 1)
            out.append(((p.detach() * g) ** 2 / norm).cpu().numpy())
    finally:
        p.requires_grad_(False)
        p.grad = None
    return np.stack(out)


def all_activity(net, movie, dt: float = DT) -> np.ndarray:
    """Activity of every node (T, n_nodes) on one stimulus, float64; initial state as in `response`."""
    import torch

    f = movie[None]
    with torch.no_grad():
        state = net.steady_state(1.0, dt, 1, grad=False)
        net.stimulus.zero(1, f.shape[1])
        net.stimulus.add_input(f)
        act = net.forward(net.stimulus(), dt, state)
    return act[0].detach().cpu().numpy().astype(np.float64)


def edge_table(net) -> dict:
    """Per edge: class (index of syn_strength), source node, weight w = sign x synapse count x scale."""
    p = net.edge_params.syn_strength
    idx = p.indices.detach().cpu().numpy()
    e = net.connectome.edges
    syn_model = net.edge_params.syn_count.semantic_values.detach().cpu().numpy()[
        net.edge_params.syn_count.indices.detach().cpu().numpy()].astype(np.float64)
    sign = np.asarray(e.sign[:], dtype=np.float64)
    s = scales(net).astype(np.float64)
    return {"cls": idx, "src": net._source_indices.detach().cpu().numpy(), "w": sign * syn_model * s[idx],
            "n_classes": len(p.keys)}


def cell_types(net) -> list[str]:
    t = net.connectome.unique_cell_types[:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in t]


def nodes(net) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(cell types, type index of every node, retinotopic column of every node)."""
    from .methods import type_and_column

    types = cell_types(net)
    n = net.connectome.nodes
    typ, column = type_and_column(n.type[:], n.u[:], n.v[:], types)
    return types, typ, column


def class_sizes(net) -> np.ndarray:
    """Class size: number of neuron-to-neuron edges of every class (needs only the connectome)."""
    p = net.edge_params.syn_strength
    return np.bincount(p.indices.detach().cpu().numpy(), minlength=len(p.keys)).astype(np.int64)


def _check_dynamics(net) -> None:
    """The gradient method rewrites the voltage update of PPNeuronIGRSynapses with relu (ensemble flow/0000)."""
    import torch
    from flyvis.network.dynamics import PPNeuronIGRSynapses

    dyn = net.dynamics
    if type(dyn) is not PPNeuronIGRSynapses or not isinstance(dyn.activation, torch.nn.ReLU):
        raise TypeError(f"unsupported dynamics {type(net.dynamics).__name__}")


@contextmanager
def scaled_currents(net, alpha):
    """Inside the block the current of every edge j of class c is w_j (1 - alpha_c) f(V_src), also in the resting
    state (steady_state uses the same update); alpha = 1 is the ablation of the truth. Same update as flyvis 1.2.0
    PPNeuronIGRSynapses otherwise."""
    import torch

    _check_dynamics(net)
    dyn = net.dynamics
    cls = net.edge_params.syn_strength.indices
    a_edge = alpha.to(cls.device)[cls]

    def write_state_velocity(vel, state, params, target_sum, x_t, dt, **kwargs):
        f = dyn.activation(state.sources.activity)
        vel.nodes.activity = (
            1 / torch.clamp(params.nodes.time_const, min=dt)
            * (-state.nodes.activity + params.nodes.bias + target_sum(params.edges.weight * ((1 - a_edge) * f))
               + x_t)
        )

    dyn.write_state_velocity = write_state_velocity
    try:
        yield
    finally:
        del dyn.write_state_velocity


def gradient_terms(net, movie, dt: float = DT) -> np.ndarray:
    """Gradient method for one stimulus, array (n_classes,): (d(u . r)/d alpha_c)^2 / ||r||^2 at alpha = 0, with
    u = r / ||r|| from the same forward pass (one forward and one backward pass). d/d alpha_c = -s_c d/d s_c, so this
    is patching with the response itself as the only output direction."""
    import torch

    n_classes = len(net.edge_params.syn_strength.keys)
    alpha = torch.zeros(n_classes, dtype=torch.float32, requires_grad=True)
    with scaled_currents(net, alpha):
        r = response(net, movie, dt, grad=True)[0]
    rd = r.detach()
    norm = float((rd * rd).sum())
    u = rd / math.sqrt(norm)
    (g,) = torch.autograd.grad((u * r).sum(), alpha)
    g = g.detach().cpu().double().numpy()
    return g * g / norm


def edge_movies(spec: dict, items: list[dict], angles: list[int] | None = None):
    """Moving-edge movies (T, 1, 721) from flyvis MovingEdge, in the order of `items`; `angles` replaces the angle
    set of `spec` (the set B' has its own angles, everything else as in B)."""
    prepare_flyvis()
    from flyvis.datasets.moving_bar import MovingEdge

    c = spec["krawedzie"]
    ds = MovingEdge(offsets=tuple(c["offsets"]), intensities=c["intensities"], speeds=c["speeds"],
                    height=c["height"], dt=c["dt"], t_pre=c["t_pre"], t_post=c["t_post"],
                    angles=list(angles or c["angles"]), device="cpu")
    out = []
    for s in items:
        key = ds.get_sequence_id_from_arguments(angle=s["kat"], width=80, intensity=s["jasnosc"],
                                                speed=s["predkosc"])
        f = ds[key].cpu()
        f = f[:, None] if f.ndim == 2 else f
        out.append(f.float())
    return out


def sintel_movies(spec: dict, sequence: str, sintel_dir: Path):
    """Sintel frames in the fly eye, as flyvis RenderedSintel and MultiTaskSintel without augmentation."""
    prepare_flyvis()
    from flyvis.datasets.augmentation.temporal import CropFrames, Interpolate
    from flyvis.datasets.rendering import BoxEye
    from flyvis.datasets.rendering.utils import split
    from flyvis.datasets.sintel_utils import load_sequence, sample_lum

    c = spec["sintel_render"]
    box = BoxEye(**c["boxfilter"])
    lum = load_sequence(Path(sintel_dir) / "training" / "final" / sequence, sample_lum, start=c["start_klatki"])
    lum_split = split(lum, box.min_frame_size[1] + 2 * box.kernel_size, c["vertical_splits"],
                      c["center_crop_fraction"])
    lum_hex = box(lum_split).cpu()
    crop = CropFrames(c["n_frames"], all_frames=False, random=False)
    interp = Interpolate(c["original_framerate"], 1 / c["dt"], mode=c["resampling"])
    return [interp(crop(lum_hex[j])) for j in range(lum_hex.shape[0])]


def load_stimuli(spec_path: Path, sintel_dir: Path | None = None) -> tuple[dict, list[tuple[str, object]]]:
    """(spec, [(id, CPU tensor)]) in the order of `stimuli_B.json`."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    return spec, movies_of(spec, spec["bodzce"], sintel_dir)


# The 40 stimuli B' of activity x weight, activity and SAE features: moving edges at 15 + 30k degrees and Sintel
# sequences not in B (disjoint from the truth). Shipped with the code, byte-identical to the file of the study.
STIMULI_BPRIME = Path(__file__).with_name("stimuli_Bprime.json")
SHA256_BPRIME = "5147e040888a9ab019db8623655a0166df37d5dd3f50e060498956ee7b571cd2"


def load_bprime(spec_b: dict, path: Path = STIMULI_BPRIME) -> dict:
    """The B' list, checked: sha256 of the study file and no stimulus shared with B."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256_BPRIME:
        raise ValueError(f"{path} differs from the B' stimuli of the study")
    bp = json.loads(raw.decode("utf-8"))
    shared = {b["id"] for b in bp["bodzce"]} & {b["id"] for b in spec_b["bodzce"]}
    if shared:
        raise ValueError(f"B' shares stimuli with B: {sorted(shared)}")
    return bp


def movies_of(spec: dict, items: list[dict], sintel_dir: Path | None = None,
              angles: list[int] | None = None) -> list[tuple[str, object]]:
    """[(id, CPU tensor)] for `items` (edges and Sintel), rendered with the settings of `spec` (stimuli_B.json)."""
    sintel_dir = Path(sintel_dir or Path(os.environ["FLYVIS_ROOT_DIR"]) / "sintel")
    edges = [b for b in items if b["typ"] == "krawedz"]
    with default_device("cpu"):
        movies = dict(zip([b["id"] for b in edges], edge_movies(spec, edges, angles), strict=True))
        cache: dict[str, list] = {}
        for b in items:
            if b["typ"] == "sintel":
                if b["sekwencja"] not in cache:
                    cache[b["sekwencja"]] = sintel_movies(spec, b["sekwencja"], sintel_dir)
                movies[b["id"]] = cache[b["sekwencja"]][b["podzial"]]
    return [(b["id"], movies[b["id"]]) for b in items]
