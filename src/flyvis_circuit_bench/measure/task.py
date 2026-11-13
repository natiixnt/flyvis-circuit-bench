"""Step `task` (v1.1, Z4): the second truth at the level of the task, from the decoded optic flow.

For every ablation e (scale of one connection class set to 0, as in the study) the trained flyvis decoder is run on
the activity of the Sintel clips of the set B and the result is compared with the ground truth flow of the clips.

Task loss, as in flyvis training (flyvis 1.2.0):

* `flyvis/solver.py`, `Solver.train`, closure `handle_batch`: `y_est = decoder[task](activity)`, then
  `task.loss(y_est, y, task)`; the total loss is the sum over tasks (here only `flow`).
* `flyvis/task/tasks.py`, `Task.loss`: `task_weight * losses[task](y_est, y) / task_weights_sum` (one task, so the
  factor is 1).
* `flyvis/task/objectives.py`, `l2norm`: `(((y_est - y_gt) ** 2).sum(dim=(1, 2, 3))).sqrt().mean()`, that is the
  root of the cumulative squared error of one clip over frames, flow components and hexals, averaged over clips.

We evaluate the same quantity per clip, `clip_losses`, so that

    T(e) = mean over clips ( loss(ablated e) - loss(intact) ),

and, as a descriptive companion that needs no ground truth,

    D(e) = mean over clips ( ||y_e - y_0|| / ||y_0|| )

(`y_0` the decoded flow of the intact model, norm over frames, components and hexals). The target is the flow of
`MultiTaskSintel` without augmentation: the first 19 frames (`CropFrames`, start 0) interpolated linearly from 24
to 50 Hz (`Interpolate`, mode linear, the training target of `transform_target`), cut into the same 3 vertical splits
and rendered with the same box filter (`ftype="sum"`) as the study's luminance clips.

The pure helpers (`clip_losses`, `flow_change`, `task_effects`) need no flyvis and are unit tested; `measure_task`
needs the `[measure]` extra.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import network as N

# ---------------------------------------------------------------- pure helpers (no flyvis)


def clip_losses(y_est: np.ndarray, y_gt: np.ndarray) -> np.ndarray:
    """flyvis `l2norm` per clip: sqrt of the summed squared error over frames, components and hexals.

    Both arrays are (clips, frames, 2, hexals); the mean of the result is the flyvis loss of the batch.
    """
    y_est, y_gt = np.asarray(y_est, dtype=np.float64), np.asarray(y_gt, dtype=np.float64)
    if y_est.shape != y_gt.shape or y_est.ndim != 4:
        raise ValueError(f"shape mismatch {y_est.shape} vs {y_gt.shape}")
    return np.sqrt(((y_est - y_gt) ** 2).sum(axis=(1, 2, 3)))


def flow_change(y_abl: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Per clip ||y_abl - y_ref|| / ||y_ref|| (norm over frames, components and hexals)."""
    y_abl, y_ref = np.asarray(y_abl, dtype=np.float64), np.asarray(y_ref, dtype=np.float64)
    if y_abl.shape != y_ref.shape or y_abl.ndim != 4:
        raise ValueError(f"shape mismatch {y_abl.shape} vs {y_ref.shape}")
    den = np.sqrt((y_ref**2).sum(axis=(1, 2, 3)))
    if (den == 0).any():
        raise ValueError("the intact decoded flow is zero for a clip")
    return np.sqrt(((y_abl - y_ref) ** 2).sum(axis=(1, 2, 3))) / den


def task_effects(decode: Callable[[], np.ndarray], ablate: Callable[[int], contextlib.AbstractContextManager],
                 n_classes: int, y_gt: np.ndarray, classes: list[int] | None = None,
                 sink: Callable[[int, np.ndarray], None] | None = None) -> dict[str, np.ndarray]:
    """Loss and flow change of every ablation.

    `decode()` returns the decoded flow (clips, frames, 2, hexals) of the network in its current state;
    `ablate(e)` is a context manager that zeroes class e and restores it. Classes not in `classes` stay nan.
    `sink(e, y)` receives the decoded flow of every ablation (and of the intact network with e = -1).
    """
    y0 = np.asarray(decode())
    ref = clip_losses(y0, y_gt)
    if sink is not None:
        sink(-1, y0)
    loss = np.full((n_classes, len(ref)), np.nan)
    change = np.full((n_classes, len(ref)), np.nan)
    for e in range(n_classes) if classes is None else classes:
        with ablate(e):
            y = np.asarray(decode())
        loss[e] = clip_losses(y, y_gt)
        change[e] = flow_change(y, y0)
        if sink is not None:
            sink(e, y)
    return {"T": (loss - ref).mean(axis=1), "D": change.mean(axis=1), "loss_ref": ref, "loss_abl": loss,
            "change": change}


# ---------------------------------------------------------------- flyvis (lazy imports)


def load_trained_with_decoder(model: str, device: str = "cpu"):
    """Trained network (checkpoint `best`) and its decoders of the same checkpoint, both in eval mode."""
    flyvis = N.prepare_flyvis()  # keep flyvis' own default device: network and decoder sample on it while created
    from flyvis.network import NetworkView

    nv = NetworkView(flyvis.results_dir / N.ENSEMBLE / model)
    raw = nv.init_network(checkpoint="best")
    decoder = nv.init_decoder(checkpoint="best")
    net = N._no_grad(raw, device)  # only now the default device is switched to `device`
    for d in decoder.values():
        d.eval()
        for p in d.parameters():
            p.requires_grad_(False)
        d.to(device)
    return net, decoder


def sintel_flows(spec: dict, sequence: str, flow_dir: Path):
    """Target flow of the 3 splits of a Sintel sequence, list of tensors (frames, 2, hexals), see the module doc."""
    N.prepare_flyvis()
    import torch
    from flyvis.datasets.augmentation.temporal import CropFrames, Interpolate
    from flyvis.datasets.rendering import BoxEye
    from flyvis.datasets.rendering.utils import split
    from flyvis.datasets.sintel_utils import load_sequence, sample_flow

    c = spec["sintel_render"]
    box = BoxEye(**c["boxfilter"])
    path = Path(flow_dir) / "training" / "flow" / sequence
    flow = load_sequence(path, sample_flow, end=c["n_frames"])  # (frames, 2, height, width), start 0
    parts = split(flow, box.min_frame_size[1] + 2 * box.kernel_size, c["vertical_splits"], c["center_crop_fraction"])
    hexed = torch.cat((box(parts[:, :, 0], ftype="sum"), box(parts[:, :, 1], ftype="sum")), dim=2).cpu()
    crop = CropFrames(c["n_frames"], all_frames=False, random=False)
    interp = Interpolate(c["original_framerate"], 1 / c["dt"], mode="linear")
    return [interp(crop(hexed[j])) for j in range(hexed.shape[0])]


def sintel_items(spec: dict) -> list[dict]:
    return [b for b in spec["bodzce"] if b["typ"] == "sintel"]


def batch_decode(net, decoder, movies, task: str = "flow") -> np.ndarray:
    """Decoded flow (clips, frames, 2, hexals) of the clips in one batch (all clips have the same length)."""
    import torch

    f = torch.stack(list(movies))
    b, t = f.shape[:2]
    with torch.no_grad():
        state = net.steady_state(1.0, N.DT, 1, grad=False)
        if b > 1:
            state = N._repeat_state(net, state, b)
        net.stimulus.zero(b, t)
        net.stimulus.add_input(f)
        act = net.forward(net.stimulus(), N.DT, state)
        return decoder[task](act).detach().cpu().double().numpy()


def measure_task(model: str, stimuli: Path, out: Path, flow_dir: Path, device: str = "cpu",
                 sintel_dir: Path | None = None, classes: list[int] | None = None,
                 store_flow: str = "float16") -> Path:
    """Measure T and D of one trained model; `classes` restricts the ablations (cost and smoke runs).

    Writes `task/C1_<model>.npz` (T, D, per clip losses and changes, clip ids, class order) and, unless
    `store_flow` is "none", `task/flow_C1_<model>.npy` with the decoded flow of every ablation, shape
    (604, clips, frames, 2, hexals), row of an ablation not measured stays 0, and `flow0` of the intact model.
    """
    import torch

    N.limit_vram(device)
    spec = json.loads(Path(stimuli).read_text(encoding="utf-8"))
    items = sintel_items(spec)
    all_movies = dict(N.movies_of(spec, items, sintel_dir))
    t0 = time.perf_counter()
    net, decoder = load_trained_with_decoder(model, device)
    cls = N.connection_classes(net)
    flows: dict[str, list] = {}
    y_gt = []
    for b in items:
        if b["sekwencja"] not in flows:
            flows[b["sekwencja"]] = sintel_flows(spec, b["sekwencja"], flow_dir)
        y_gt.append(flows[b["sekwencja"]][b["podzial"]].numpy())
    y_gt = np.stack(y_gt)
    movies = [all_movies[b["id"]].to(device) for b in items]
    if y_gt.shape[1] != movies[0].shape[0]:
        raise ValueError(f"target has {y_gt.shape[1]} frames, the stimulus {movies[0].shape[0]}")
    out_dir = Path(out) / "task"
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = None
    flow0 = {}
    dtype = np.float16 if store_flow == "float16" else np.float32

    def decode():
        return batch_decode(net, decoder, movies)

    def sink(e: int, y: np.ndarray) -> None:
        nonlocal mm
        if store_flow == "none":
            return
        if e < 0:
            flow0["y"] = y.astype(dtype)
            return
        if mm is None:
            mm = np.lib.format.open_memmap(out_dir / f"flow_C1_{model}.npy", mode="w+", dtype=dtype,
                                           shape=(len(cls), *y.shape))
        mm[e] = y.astype(dtype)

    res = task_effects(decode, lambda e: N.ablated(net, e), len(cls), y_gt, classes, sink)
    if mm is not None:
        mm.flush()
    peak = float(torch.cuda.max_memory_allocated() / 2**20) if str(device).startswith("cuda") else float("nan")
    meta = {"condition": "C1", "model": model, "device": device, "store_flow": store_flow,
            "classes_measured": len(cls) if classes is None else len(classes),
            "loss": "flyvis l2norm per clip, flow task, ablated minus intact (see measure/task.py)",
            "target": "MultiTaskSintel flow, no augmentation, linear interpolation 24 to 50 Hz",
            "seconds": time.perf_counter() - t0, "peak_vram_mib": peak}
    path = out_dir / f"C1_{model}.npz"
    np.savez(path, T=res["T"], D=res["D"], loss_ref=res["loss_ref"], loss_abl=res["loss_abl"],
             change=res["change"], bodzce=np.array([b["id"] for b in items]),
             zrodlo=np.array([c["source"] for c in cls]), cel=np.array([c["target"] for c in cls]),
             flow0=flow0.get("y", np.zeros(0)), meta=json.dumps(meta))
    return path
