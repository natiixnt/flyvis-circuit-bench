"""v1.1 Z7 (exploratory): attribution patching with a second-order (Hessian) correction on the diagonal.

Definition: `docs/analysis-plan-v1.1.md`, section Z7 (clarification of 2026-11-13). The scale of class c is
s_c (1 - alpha_c) (see `network.scaled_currents`), the ablation of the truth is alpha_c = 1, patching is the first-order
change along alpha_c. With `f_k(alpha) = v_k . r(alpha)`:

    first_kc  = d f_k / d alpha_c at 0          (= -s_c g_kc, the patching change)
    second_kc = d^2 f_k / d alpha_c^2 at 0      (= s_c^2 H_kc,c, the diagonal of the Hessian)
    change_kc = first_kc + 0.5 second_kc        (Taylor expansion of the ablation to second order)
    term_kc   = change_kc^2 / ||r||^2           (patching: first_kc^2 / ||r||^2)

Two exact backends compute `first` and `second` for every readout direction k and class c:

* `reverse`: the Hessian-vector product with the unit vector e_c by double backward (torch.autograd), one per
  (direction, class); the reference definition of the plan.
* `forward`: the second directional derivative of the response vector r along e_c by nested forward mode
  (torch.func.jvp of jvp), one per class for all directions at once, since `d^2 f_k / d alpha_c^2 = v_k . d^2 r / d
  alpha_c^2`. Same numbers, K times fewer passes.

The functions take `respond(alpha) -> r` (any differentiable map from the class vector alpha, shape (n,), to the
response r, shape (T, n_types)), so they are tested on small synthetic networks without flyvis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import network as N

BACKENDS = ("forward", "reverse")
SUBSET_SEED = 20261101
SUBSET_SIZE = 8  # plan, Z7, second clarification of 2026-11-13: 8 of the 40 stimuli of B


def select_stimuli(n_total: int, n: int = SUBSET_SIZE, seed: int = SUBSET_SEED) -> list[int]:
    """The preregistered subset: sorted indices into the order of stimuli_B.json (4, 12, 16, 18, 20, 27, 37, 39
    for 8 of 40 and the study seed)."""
    return sorted(int(i) for i in np.random.default_rng(seed).choice(n_total, size=n, replace=False))


def second_order_reverse(respond, n: int, v, classes=None) -> dict:
    """Double backward: for every direction k the gradient g_k = d f_k / d alpha with the graph kept, then one
    backward pass per class c through g_k[c] (the component c of the Hessian-vector product H e_c)."""
    import torch

    classes = list(range(n)) if classes is None else list(classes)
    alpha = torch.zeros(n, dtype=v.dtype, requires_grad=True)
    r = respond(alpha)
    k_dir = v.shape[0]
    first = np.full((k_dir, n), np.nan)
    second = np.full((k_dir, n), np.nan)
    for k in range(k_dir):
        (g,) = torch.autograd.grad((v[k].to(r.device) * r).sum(), alpha, create_graph=True, retain_graph=True)
        first[k] = g.detach().cpu().double().numpy()
        for c in classes:
            (h,) = torch.autograd.grad(g[c], alpha, retain_graph=True)
            second[k, c] = float(h[c])
    rd = r.detach().double()
    return {"first": first, "second": second, "norm": float((rd * rd).sum())}


def second_order_forward(respond, n: int, v, classes=None) -> dict:
    """Nested forward mode along e_c: r'(c) and r''(c), then dot with every direction."""
    import torch
    from torch.func import jvp

    classes = list(range(n)) if classes is None else list(classes)
    first = np.full((v.shape[0], n), np.nan)
    second = np.full((v.shape[0], n), np.nan)
    norm = None
    one = torch.ones((), dtype=v.dtype)
    t0 = torch.zeros((), dtype=v.dtype)
    vf = v.reshape(v.shape[0], -1).double()
    for c in classes:
        e = torch.zeros(n, dtype=v.dtype)
        e[c] = 1.0

        def inner(t, e=e):
            return jvp(lambda s: respond(s * e), (t,), (one,))

        (r, r1), (_, r2) = jvp(inner, (t0,), (one,))
        if norm is None:
            rd = r.detach().double()
            norm = float((rd * rd).sum())
        first[:, c] = (vf.to(r1.device) @ r1.detach().double().reshape(-1)).cpu().numpy()
        second[:, c] = (vf.to(r2.device) @ r2.detach().double().reshape(-1)).cpu().numpy()
    return {"first": first, "second": second, "norm": norm}


def second_order(respond, n: int, v, classes=None, backend: str = "forward") -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} not in {BACKENDS}")
    fn = second_order_forward if backend == "forward" else second_order_reverse
    return fn(respond, n, v, classes)


def terms(res: dict) -> dict:
    """Per direction and class (K, n): patching term, corrected term, and the parts of the correction."""
    f, s, norm = res["first"], res["second"], res["norm"]
    ch = f + 0.5 * s
    return {"P": f * f / norm, "PH": ch * ch / norm, "cross": f * (0.5 * s) / norm, "Q": (0.5 * s) ** 2 / norm}


# ---------------------------------------------------------------------------------------------------- flyvis


def responder(net, movie, dt: float = N.DT):
    """respond(alpha) for one stimulus of a flyvis network: central-column response (T, n_types)."""

    def respond(alpha):
        with N.scaled_currents(net, alpha):
            return N.response(net, movie, dt, grad=True)[0]

    return respond


def stimulus_terms(net, movie, v, classes=None, backend: str = "forward") -> dict:
    n = len(net.edge_params.syn_strength.keys)
    res = second_order(responder(net, movie), n, v, classes, backend)
    return {k: x.mean(axis=0) for k, x in terms(res).items()} | {"norm": res["norm"]}


def measure_hvp(model: str, stimuli: Path, out: Path, device: str = "cpu", k: int = 32, backend: str = "forward",
                classes: list[int] | None = None, max_stimuli: int | None = None, work: Path | None = None,
                n_stimuli: int | None = None) -> Path:
    """P+HVP of one trained model on the stimuli of B. Per-stimulus state files in `work` make the run resumable
    (RESUME = run again with the same arguments). `n_stimuli` takes the preregistered random subset of that size
    (`select_stimuli`); `classes` and `max_stimuli` (the first N stimuli) restrict it for smoke and cost runs."""
    import torch

    from . import run

    N.limit_vram(device)
    spec, items = N.load_stimuli(stimuli)
    ids = [i for i, _ in items]
    b1, b2 = [ids.index(i) for i in spec["B1"]], [ids.index(i) for i in spec["B2"]]
    work = Path(work) if work else Path(out) / "hvp" / f"_state_C1_{model}"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    net = N.load_trained_on_flyvis_device(model, device)
    n_classes = len(N.connection_classes(net))
    n_types = len(N.central_cells(net))
    if n_stimuli is not None:
        todo = select_stimuli(len(items), n_stimuli)
    else:
        todo = list(range(len(items) if max_stimuli is None else min(max_stimuli, len(items))))
    parts = ("P", "PH", "cross", "Q")
    for b in todo:
        state = work / f"b{b:03d}.npz"
        if state.is_file():
            continue
        movie = items[b][1].to(device)
        v = torch.from_numpy(N.directions(k, int(movie.shape[0]), n_types, b))
        tb = time.perf_counter()
        res = stimulus_terms(net, movie, v, classes, backend)
        np.savez(state, **{p: res[p] for p in parts}, norm=res["norm"], seconds=time.perf_counter() - tb)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    run.torch_cpu()
    done = [b for b in todo if (work / f"b{b:03d}.npz").is_file()]
    stack = {p: np.stack([np.load(work / f"b{b:03d}.npz")[p] for b in done]) for p in parts}  # (stimuli, classes)
    sec = [float(np.load(work / f"b{b:03d}.npz")["seconds"]) for b in done]
    d1, d2 = [b for b in done if b in b1], [b for b in done if b in b2]
    sel = {b: i for i, b in enumerate(done)}
    meta = {"condition": "C1", "model": model, "device": device, "K": k, "backend": backend,
            "n_classes": n_classes, "n_stimuli": len(done), "stimuli_index": done,
            "classes_measured": None if classes is None else len(classes),
            "seconds_total": time.perf_counter() - t0, "seconds_per_stimulus_mean": float(np.mean(sec)),
            "definition": "docs/analysis-plan-v1.1.md Z7 (2026-11-13)"}
    if str(device).startswith("cuda"):
        meta["max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    path = Path(out) / "hvp" / f"C1_{model}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    def half(arr, idx):
        return arr[[sel[b] for b in idx]].mean(axis=0) if idx else np.full(arr.shape[1], np.nan)

    np.savez(path, PH=stack["PH"].mean(axis=0), PH_B1=half(stack["PH"], d1), PH_B2=half(stack["PH"], d2),
             P_chk=stack["P"].mean(axis=0), cross=stack["cross"].mean(axis=0), Q=stack["Q"].mean(axis=0),
             PH_b=stack["PH"].astype(np.float32), bodzce=np.array([ids[b] for b in done]), meta=json.dumps(meta))
    return path
