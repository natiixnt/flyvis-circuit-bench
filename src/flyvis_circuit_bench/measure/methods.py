"""The three activity-based methods with learned weights; the parts that need no flyvis and no torch (numpy)

The network side (all-node activity, the gradient method) is in `network.py`, the file writers in `run.py`.
Definitions, as in the study (docs/methodology.md):

- |weights| x neuron activity (`M3`): sum over the edges j of a class of |w_j| * mean_b mean_t relu(V_src(j)),
  over the 40 stimuli of B' (disjoint from the truth stimuli B);
- |weights| x source activity (`M_akt`): sum of |weights| of the class x mean |activity| of the neurons of its
  source type, over every time step of B';
- |weights| x source SAE code (`M5`): sum of |weights| of the class x mean L1 norm of the sparse-autoencoder code
  of the neurons of its source type. The SAE input is one row per neuron: mean, std, max and min of its
  activity over time on each of the 40 stimuli of B' (160 numbers), standardized with the training neurons.
  Configuration: `SAEConfig`.

The formulas keep the operation order of the study code, so the same activities give the same bits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

SEED = 20261101


@dataclass(frozen=True)
class SAEConfig:
    """Everything that defines the SAE of the `M5` method (the defaults are the published configuration)."""

    dictionary_size: int = 640  # 4 statistics x 160 (number of features of the code)
    lambda_grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1)  # L1 sparsity penalty, chosen per model
    max_fvu: float = 0.10  # largest lambda whose validation FVU is <= this; else the smallest lambda
    epochs: int = 50
    minibatch: int = 1024
    learning_rate: float = 1e-3  # Adam, beta (0.9, 0.999), eps 1e-8; decoder columns renormalized after each step
    validation_fraction: float = 0.2  # share of retinotopic columns held out (all neurons of a column together)
    seed: int = SEED  # split, initialization and minibatch order
    training_data: str = ("per neuron: mean, std, max, min of activity over time on each of the 40 stimuli of B' "
                          "(160 features), z-scored with the training neurons; cell-type labels are not used")

    def to_dict(self) -> dict:
        return asdict(self)


SAE = SAEConfig()


def relu(x):
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------- |weights| x neuron activity (M3)


class ActivityAccumulator:
    """Streaming mean relu activity per neuron (f-bar) and |weights| x neuron activity, one stimulus at a time.

    w: weight of every edge (sign x synapse count x scale), src: source node of every edge, cls: class of every
    edge. Every stimulus has the same weight; stimuli of the truth set B must not be added (`skip`)."""

    def __init__(self, w, src, cls, n_classes: int, skip: Sequence[str] = ()):
        self.w, self.src, self.cls = np.asarray(w, float), np.asarray(src), np.asarray(cls)
        self.n_classes, self.skip = n_classes, set(skip)
        self.f_sum = None
        self.stimuli: list[str] = []

    def add(self, ident: str, act: np.ndarray) -> None:
        """act: activity of every node (T, n_nodes)."""
        if ident in self.skip:
            raise ValueError(f"stimulus {ident} belongs to the truth set B")
        if ident in self.stimuli:
            raise ValueError(f"stimulus {ident} added twice")
        f = relu(np.asarray(act, dtype=np.float64)).mean(axis=0)
        self.f_sum = f if self.f_sum is None else self.f_sum + f
        self.stimuli.append(ident)

    def result(self) -> dict:
        if not self.stimuli:
            raise ValueError("no stimuli")
        fbar = self.f_sum / len(self.stimuli)
        return {"M3": activity_x_weight(self.w, self.src, self.cls, fbar, self.n_classes), "fbar": fbar}


def activity_x_weight(w, src, cls, fbar, n_classes: int) -> np.ndarray:
    """M3(e) = sum over edges j of e of |w_j| * f-bar of the source neuron of j."""
    return np.bincount(cls, weights=np.abs(w) * fbar[src], minlength=n_classes)


# ---------------------------------------------------------------- neuron statistics


def neuron_statistics(act: np.ndarray) -> np.ndarray:
    """(n_nodes, 4): mean, std, max, min over time of one stimulus' activity (T, n_nodes)."""
    a = np.asarray(act, dtype=np.float64)
    return np.stack([a.mean(axis=0), a.std(axis=0), a.max(axis=0), a.min(axis=0)], axis=1)


class StatisticsAccumulator:
    """SAE input (n_nodes, 4 x n_stimuli) and mean |activity| per neuron over all time steps, streaming."""

    def __init__(self):
        self.columns: list[np.ndarray] = []
        self.abs_sum, self.steps = 0.0, 0
        self.stimuli: list[str] = []

    def add(self, ident: str, act: np.ndarray) -> None:
        a = np.asarray(act, dtype=np.float64)
        self.columns.append(neuron_statistics(a))
        self.abs_sum = self.abs_sum + np.abs(a).sum(axis=0)
        self.steps += len(a)
        self.stimuli.append(ident)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.stimuli:
            raise ValueError("no stimuli")
        return np.concatenate(self.columns, axis=1), self.abs_sum / self.steps


def type_and_column(node_types, u, v, types: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Index of the cell type (in the order of `types`) and of the retinotopic column (u, v) of every node."""
    idx = {t: i for i, t in enumerate(types)}
    typ = np.array([idx[t.decode() if isinstance(t, bytes) else str(t)] for t in node_types])
    uv = np.stack([np.asarray(u), np.asarray(v)], axis=1)
    _, column = np.unique(uv, axis=0, return_inverse=True)
    return typ, column.ravel()


def type_mean(value: np.ndarray, typ: np.ndarray, n_types: int) -> np.ndarray:
    count = np.bincount(typ, minlength=n_types).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.bincount(typ, weights=value, minlength=n_types) / count


def class_from_type(abs_weight: np.ndarray, source_type: np.ndarray, per_type: np.ndarray) -> np.ndarray:
    """score(e) = sum of |weights| of e x value of its source type (used by M_akt and M5)."""
    return np.asarray(abs_weight, dtype=float) * np.asarray(per_type, dtype=float)[np.asarray(source_type)]


# ---------------------------------------------------------------- SAE (numpy, Adam)


def standardize(x: np.ndarray, train: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-score with the mean and std of the training rows; a constant feature becomes 0."""
    x = np.asarray(x, dtype=np.float64)
    xu = x if train is None else x[train]
    mu = xu.mean(axis=0)
    sd = xu.std(axis=0)
    sd_nonzero = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd_nonzero * (sd > 0), mu, sd


def split(n: int, seed: int = SEED, fraction: float = SAE.validation_fraction) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(fraction * n))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def column_split(column: np.ndarray, seed: int = SEED, fraction: float = SAE.validation_fraction):
    """Training and validation neurons: a random `fraction` of the columns (with all their neurons) validates."""
    column = np.asarray(column)
    cols = np.unique(column)
    _, val_c = split(len(cols), seed, fraction)
    val = np.isin(column, cols[val_c])
    return np.flatnonzero(~val), np.flatnonzero(val)


def _normalize_columns(w: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(w, axis=0, keepdims=True)
    return w / np.where(n > 0, n, 1.0)


def sae_init(d: int, m: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    w_d = _normalize_columns(rng.standard_normal((d, m)))
    return {"W_e": w_d.T.copy(), "b_e": np.zeros(m), "W_d": w_d, "b_d": np.zeros(d)}


def sae_forward(p: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, z, x_hat): a = W_e (x - b_d) + b_e, z = relu(a), x_hat = W_d z + b_d."""
    a = (x - p["b_d"]) @ p["W_e"].T + p["b_e"]
    z = np.maximum(a, 0.0)
    return a, z, z @ p["W_d"].T + p["b_d"]


def sae_loss(p: dict, x: np.ndarray, lam: float) -> float:
    _, z, xh = sae_forward(p, x)
    return float(np.mean((xh - x) ** 2) + lam * np.mean(np.sum(z, axis=1)))


def sae_gradient(p: dict, x: np.ndarray, lam: float) -> dict:
    """Gradient of the loss: mean squared error over elements + lambda x mean L1 norm of the code over rows."""
    b, d = x.shape
    xt = x - p["b_d"]
    a, z, xh = sae_forward(p, x)
    dxh = 2.0 * (xh - x) / (b * d)
    dz = dxh @ p["W_d"] + lam / b
    da = dz * (a > 0)
    dxt = da @ p["W_e"]
    return {
        "W_d": dxh.T @ z,
        "b_d": dxh.sum(axis=0) - dxt.sum(axis=0),
        "W_e": da.T @ xt,
        "b_e": da.sum(axis=0),
    }


def train_sae(x: np.ndarray, lam: float, cfg: SAEConfig = SAE) -> dict:
    """Adam on minibatches, decoder columns projected to norm 1 after every step; `cfg.seed` fixes everything."""
    x = np.asarray(x, dtype=np.float64)
    p = sae_init(x.shape[1], cfg.dictionary_size, cfg.seed)
    p["b_d"] = x.mean(axis=0)
    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    rng = np.random.default_rng([cfg.seed, 1])
    b1, b2, eps, t, lr = 0.9, 0.999, 1e-8, 0, cfg.learning_rate
    for _ in range(cfg.epochs):
        perm = rng.permutation(len(x))
        for s in range(0, len(x), cfg.minibatch):
            g = sae_gradient(p, x[perm[s : s + cfg.minibatch]], lam)
            t += 1
            for k in p:
                m1[k] = b1 * m1[k] + (1 - b1) * g[k]
                m2[k] = b2 * m2[k] + (1 - b2) * g[k] ** 2
                p[k] = p[k] - lr * (m1[k] / (1 - b1**t)) / (np.sqrt(m2[k] / (1 - b2**t)) + eps)
            p["W_d"] = _normalize_columns(p["W_d"])
    return p


def encode(p: dict, x: np.ndarray) -> np.ndarray:
    return sae_forward(p, np.asarray(x, dtype=np.float64))[1]


def fvu(p: dict, x: np.ndarray) -> float:
    """Fraction of variance unexplained: sum (x_hat - x)^2 / sum (x - mean)^2."""
    x = np.asarray(x, dtype=np.float64)
    xh = sae_forward(p, x)[2]
    denom = float(np.sum((x - x.mean(axis=0)) ** 2))
    return float(np.sum((xh - x) ** 2)) / denom if denom > 0 else math.nan


def select_lambda(x: np.ndarray, train: np.ndarray, val: np.ndarray, cfg: SAEConfig = SAE) -> dict:
    """One SAE per lambda on the training neurons, FVU on the validation neurons; the largest lambda with FVU <=
    `cfg.max_fvu`, otherwise the smallest (flagged). Cell-type labels take no part."""
    fits = []
    for lam in cfg.lambda_grid:
        p = train_sae(x[train], lam, cfg)
        fits.append({"lambda": float(lam), "FVU": fvu(p, x[val]), "p": p})
    ok = [f for f in fits if f["FVU"] <= cfg.max_fvu]
    best = max(ok, key=lambda f: f["lambda"]) if ok else min(fits, key=lambda f: f["lambda"])
    return {"lambda": best["lambda"], "FVU": best["FVU"], "p": best["p"], "meets_max_fvu": bool(ok),
            "FVU_grid": [f["FVU"] for f in fits]}


def code_l1_per_type(z: np.ndarray, typ: np.ndarray, n_types: int) -> np.ndarray:
    """g_t = mean L1 norm of the code over the neurons of type t."""
    l1 = np.abs(np.asarray(z, dtype=np.float64)).sum(axis=1)
    return type_mean(l1, typ, n_types)


def activity_and_sae(x_raw: np.ndarray, abs_neuron: np.ndarray, typ, column, n_types: int, abs_weight,
                     source_type, cfg: SAEConfig = SAE) -> dict:
    """M_akt and M5 of one network from the SAE input (n_nodes, 4 x n_stimuli) and the mean |activity| (n_nodes,).

    Keys as in `activity_sae/C1_<model>.npz` of the measurement package."""
    typ = np.asarray(typ)
    train, val = column_split(column, cfg.seed, cfg.validation_fraction)
    x, mu, sd = standardize(x_raw, train)
    fit = select_lambda(x, train, val, cfg)
    g = code_l1_per_type(encode(fit["p"], x), typ, n_types)
    a_type = type_mean(np.asarray(abs_neuron, dtype=np.float64), typ, n_types)
    return {
        "M5": class_from_type(abs_weight, source_type, g),
        "M_akt": class_from_type(abs_weight, source_type, a_type),
        "g": g,
        "a_typu": a_type,
        "lambda": np.asarray(fit["lambda"]),
        "FVU": np.asarray(fit["FVU"]),
        "FVU_siatka": np.asarray(fit["FVU_grid"]),
        "mu": mu,
        "sd": sd,
        "n_uczace": np.asarray(len(train)),
        "n_walidacja": np.asarray(len(val)),
        "meets_max_fvu": fit["meets_max_fvu"],
    }
