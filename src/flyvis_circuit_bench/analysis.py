"""All result tables of the benchmark, computed from the measurement files.

Sections (same order as `docs/results.md`):

1. `week1`: methods S, A, P and the truth ceiling in C1 (trained) and C3 (initialization), N = 5, plus the
   preregistered contrasts and the failure criterion.
2. `mechanism`: the week 2 follow-up. Delta S and D_C1 - D_C3 on AUROC and on the non-saturating measure c1.
3. `replication`: the week 5 primary family T5.1-T5.4 (pooled N = 10, original 000-004, replication 005-009),
   one-sided p values with Holm correction, decisions and their combination with weeks 1-4.
4. `method_means`: mean AUROC of every method in C1 and on the second stimulus family F2, N = 10.
5. `controls`: validity controls K1-K5 on the original five models.
6. `leaderboard`: every method on AUROC and on three metrics added after the preregistration (share of the
   achievable signal, AUPRC, precision@61), for all 10 models and for each half; `readme_example`: the
   example method of the README ("Add your method") on the same metrics.

Every rule (thresholds, sign rule, Holm level, undetermined rule) was fixed in the preregistration before the
corresponding data existed; see `docs/preregistration.md`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

import numpy as np

from . import metrics as M
from .data import CONDITIONS, MODELS_ORIGINAL, MODELS_POOLED, MODELS_REPLICATION
from .stats import NEGATIVE, POSITIVE, UNDETERMINED, Summary, holm, one_sided_p, summarize

VERSIONS = {"pooled": MODELS_POOLED, "original": MODELS_ORIGINAL, "replication": MODELS_REPLICATION}
PRIMARY = ("T5.1", "T5.2", "T5.3", "T5.4")
CEILING_MARGIN = 0.05  # failure criterion 2: AUROC(S) within 0.05 of the ceiling
CEILING_TOL = 1e-9  # an exact 0.05 counts as failure
HOLM_ALPHA = 0.025  # one-sided
MDE = {10: 0.07, 5: 0.12}  # "no effect larger than MDE" label for inconclusive results
F2_MIN_POSITIVE = 0.5  # (i) at least 50% of classes with E_F2 > 0
F2_MIN_SPREAD = 3.0  # (ii) q90 / median of positive E_F2
F2_MIN_CEILING = 0.95  # (iii) AUROC(E_F2a against truth from E_F2b)
F2_MIN_BARS = 20  # (iv) bars with nonzero response
PHOTORECEPTORS = tuple(f"R{i}" for i in range(1, 9))
C3_SCALE = 0.01  # flyvis SynapseCountScaling rho in the ensemble config
K5_THRESHOLD = 1e-4

SURVIVES, REFUTED, INCONCLUSIVE = "survives", "refuted", "inconclusive"
CHEAP_SURVIVE, PARTIAL = "cheap methods survive", "partial"

# ---------------------------------------------------------------- week 1


def file_metrics(d: dict) -> dict:
    """Metrics of one ablation file: every method against the truth from E on B, ceiling from halves B1/B2."""
    e = d["E_B"]
    out: dict = {"n_positive": {"B": int(M.binary_truth(e).sum()), "B1": int(M.binary_truth(d["E_B1"]).sum()),
                                "B2": int(M.binary_truth(d["E_B2"]).sum())}}
    for method in ("S", "A", "P"):
        out[method] = {mi: M.measure(mi, d[method], e) for mi in M.MEASURES}
    out["ceiling"] = {mi: M.measure(mi, d["E_B1"], d["E_B2"]) for mi in M.MEASURES}
    return out


def _per_model(mm: dict[str, dict], f: Callable[[dict], float]) -> dict[str, float]:
    out = {}
    for m in sorted(mm):
        try:
            out[m] = float(f(mm[m]))
        except KeyError:
            continue
    return out


def _row(key: str, label: str, s: Summary) -> dict:
    return {"key": key, "label": label, **s.to_dict()}


def week1(data: dict) -> dict:
    models = data["models"]
    mm = {c: {m: file_metrics(models[m][f"B_{c}"]) for m in MODELS_ORIGINAL if f"B_{c}" in models[m]}
          for c in CONDITIONS}
    methods = []
    for c in CONDITIONS:
        for method in ("S", "A", "P", "ceiling"):
            for mi in ("AUROC", "AUPRC", "Spearman"):
                s = summarize(_per_model(mm[c], lambda x, me=method, mi=mi: x[me][mi]))
                methods.append(_row(f"{c}|{method}|{mi}", f"{c} {method} {mi}", s))
    contrasts = []
    for c in CONDITIONS:
        for a, b, mi in (("P", "S", "AUROC"), ("P", "A", "AUROC"), ("P", "S", "AUPRC"), ("P", "A", "AUPRC")):
            s = summarize(_per_model(mm[c], lambda x, a=a, b=b, mi=mi: x[a][mi] - x[b][mi]))
            name = f"D_{c}" if (a, b, mi) == ("P", "S", "AUROC") else f"{mi}(P) - {mi}({b}), {c}"
            contrasts.append(_row(f"{c}|{a}-{b}|{mi}", name, s))
    both = sorted(set(mm["C1"]) & set(mm["C3"]))
    for b in ("S", "A"):
        diff = {m: (mm["C1"][m]["P"]["AUROC"] - mm["C1"][m][b]["AUROC"])
                - (mm["C3"][m]["P"]["AUROC"] - mm["C3"][m][b]["AUROC"]) for m in both}
        contrasts.append(_row(f"DD|P-{b}|AUROC", f"D_C1 - D_C3 (AUROC, P - {b})", summarize(diff)))
    return {"per_model": mm, "methods": methods, "contrasts": contrasts,
            "failure_criterion": failure_criterion(mm["C1"])}


def failure_criterion(m1: dict[str, dict]) -> dict:
    """Preregistered failure of W1: (1) a lower CI bound of D_C1 <= 0, or (2) AUROC(S) within 0.05 of ceiling."""
    d = summarize({m: m1[m]["P"]["AUROC"] - m1[m]["S"]["AUROC"] for m in m1})
    s = float(np.mean([m1[m]["S"]["AUROC"] for m in sorted(d.per_model)]))
    ceiling = float(np.mean([m1[m]["ceiling"]["AUROC"] for m in sorted(d.per_model)]))
    cond1 = not (d.boot[0] > 0 and d.t[0] > 0)
    cond2 = (ceiling - s) <= CEILING_MARGIN + CEILING_TOL
    return {"D_C1": d.to_dict(), "auroc_S": s, "auroc_ceiling": ceiling, "ceiling_minus_S": ceiling - s,
            "condition_1_lower_bound_not_positive": bool(cond1), "condition_2_S_near_ceiling": bool(cond2),
            "failed": bool(cond1 or cond2)}


# ---------------------------------------------------------------- week 2: mechanism


def mechanism(w1: dict) -> list[dict]:
    m1, m3 = w1["per_model"]["C1"], w1["per_model"]["C3"]
    both = sorted(set(m1) & set(m3))
    rows = []
    for mi in ("AUROC", "c1"):
        ds = {m: m1[m]["S"][mi] - m3[m]["S"][mi] for m in both}
        dd = {m: (m1[m]["P"][mi] - m1[m]["S"][mi]) - (m3[m]["P"][mi] - m3[m]["S"][mi]) for m in both}
        rows.append(_row(f"dS|{mi}", f"Delta S = {mi}(S, C1) - {mi}(S, C3)", summarize(ds)))
        rows.append(_row(f"DD|{mi}", f"D_C1 - D_C3 = Delta P - Delta S ({mi})", summarize(dd)))
    return rows


# ---------------------------------------------------------------- week 5: replication


def f2_validity(d: dict) -> dict:
    """Validity conditions (i)-(iv) of the second stimulus family for one model (fixed before any E_F2)."""
    e = np.asarray(d["E_F2"], dtype=np.float64)
    n = len(e)
    pos = e[np.isfinite(e) & (e > 0)]
    i_ok = pos.size >= math.ceil(F2_MIN_POSITIVE * n)
    if pos.size:
        median = float(np.median(pos))
        spread = float(np.quantile(pos, 0.9) / median) if median > 0 else float("nan")
    else:
        spread = float("nan")
    finite_max = bool(n and np.isfinite(e).all() and np.isfinite(np.max(e)))
    ii_ok = bool(finite_max and math.isfinite(spread) and spread >= F2_MIN_SPREAD)
    ea, eb = np.asarray(d["E_F2a"], np.float64), np.asarray(d["E_F2b"], np.float64)
    ceiling = (M.auroc(ea, M.binary_truth(eb)) if np.isfinite(ea).all() and np.isfinite(eb).all()
               else float("nan"))
    iii_ok = bool(math.isfinite(ceiling) and ceiling >= F2_MIN_CEILING)
    r = np.asarray(d["r_ref_norma_F2"], dtype=np.float64)
    bars = int((np.isfinite(r) & (r > 0)).sum())
    iv_ok = bars >= F2_MIN_BARS
    return {"i_classes_E_positive": int(pos.size), "ii_q90_over_median": spread, "ii_max_finite": finite_max,
            "iii_ceiling_auroc": ceiling, "iv_nonzero_bars": bars,
            "valid": bool(i_ok and ii_ok and iii_ok and iv_ok)}


def c1_scores(d: dict, size: np.ndarray | None) -> dict[str, np.ndarray]:
    """Scores of every method of one trained model (only the files present)."""
    b = d["B_C1"]
    scores = {k: b[k] for k in ("S", "A", "P")}
    if "M13" in d:
        scores.update({"M1": d["M13"]["M1"], "M3": d["M13"]["M3"]})
    if "M5" in d:
        scores.update({"M_akt": d["M5"]["M_akt"], "M5": d["M5"]["M5"]})
    if size is not None:
        scores["size"] = size
    return scores


def model_metrics(d: dict, size: np.ndarray | None) -> dict:
    """AUROC of every method of one model in C1, C3 and on F2 (only the files present; F2 only if valid)."""
    out: dict = {}
    b = d.get("B_C1")
    if b is not None:
        e = b["E_B"]
        out["C1"] = {k: M.measure("AUROC", v, e) for k, v in c1_scores(d, size).items()}
        out["C1"]["ceiling"] = M.measure("AUROC", b["E_B1"], b["E_B2"])
    b3 = d.get("B_C3")
    if b3 is not None:
        out["C3"] = {k: M.measure("AUROC", b3[k], b3["E_B"]) for k in ("S", "A", "P")}
    f2 = d.get("F2")
    if f2 is not None:
        out["F2_validity"] = f2_validity(f2)
        if out["F2_validity"]["valid"]:
            e2 = f2["E_F2"]
            scores = {"S": f2["S"], "A": f2["A"], "P_F2": f2["P_F2"], "M1_F2": f2["M1_F2"]}
            if size is not None:
                scores["size"] = size
            out["F2"] = {k: M.measure("AUROC", v, e2) for k, v in scores.items()}
            out["F2"]["ceiling"] = M.measure("AUROC", f2["E_F2a"], f2["E_F2b"])
    return out


def quantity(per_model: dict[str, float], plan: tuple[str, ...], direction: int = 1) -> dict:
    """Mean with both CIs, sign rule, undetermined rule (>= half of the plan omitted) and one-sided p."""
    s = summarize(per_model, plan)
    undetermined = len(s.omitted) >= math.ceil(len(plan) / 2)
    sign = UNDETERMINED if undetermined else s.sign()
    mde = MDE.get(len(plan), float("nan"))
    bounds = (*s.boot, *s.t)
    return {**s.to_dict(), "sign": sign, "undetermined": bool(undetermined),
            "p_one_sided": one_sided_p(list(s.per_model.values()), direction),
            "no_effect_above_mde": bool(sign == "inconclusive"
                                        and all(math.isfinite(g) and abs(g) <= mde for g in bounds))}


def decide_t51(q: dict, ceiling_minus_s: float, holm_ok: bool | None) -> str:
    """T5.1 and T5.4: survives (both CIs > 0, ceiling - S > 0.05, Holm), refuted (both upper bounds < 0)."""
    if q["undetermined"]:
        return UNDETERMINED
    if q["sign"] == POSITIVE and ceiling_minus_s > CEILING_MARGIN + CEILING_TOL and holm_ok is not False:
        return SURVIVES
    if q["sign"] == NEGATIVE:
        return REFUTED
    return INCONCLUSIVE


def decide_t52(q1: dict, q3: dict, holm_ok: bool | None) -> str:
    if q1["undetermined"] or q3["undetermined"]:
        return UNDETERMINED
    pos = [q["sign"] == POSITIVE for q in (q1, q3)]
    neg = [q["sign"] == NEGATIVE for q in (q1, q3)]
    if all(pos):
        return CHEAP_SURVIVE if holm_ok is not False else INCONCLUSIVE
    if sum(pos) == 1:
        return PARTIAL
    if not any(pos) and any(neg):
        return REFUTED
    return INCONCLUSIVE


def decide_t53(q: dict, holm_ok: bool | None) -> str:
    if q["undetermined"]:
        return UNDETERMINED
    if q["sign"] == NEGATIVE and holm_ok is not False:
        return SURVIVES
    if q["sign"] == POSITIVE:
        return REFUTED
    return INCONCLUSIVE


_SIGN = {POSITIVE: 1, NEGATIVE: -1, "inconclusive": 0, SURVIVES: 1, REFUTED: -1, CHEAP_SURVIVE: 1,
         PARTIAL: 0}


def combine(d5: str | None, d10: str | None, dn: str | None) -> str:
    """Combination rule with weeks 1-4 (fixed in the week 5 annex), on the signs of the three decisions."""
    s5, s10, sn = (None if x is None else _SIGN.get(x) for x in (d5, d10, dn))
    if s10 is None:
        return "undetermined (no effect on the main result)"
    if sn is not None and s10 != 0 and sn == -s10:
        return "disputed (disagrees with replication)"
    if s5 is not None and sn is not None and s10 != s5 and sn == s5:
        return "disputed (disagrees with replication)"
    if s5 is None:
        return "no discovery decision (the decision on all 10 is adopted)"
    if s10 == s5:
        return "confirmed"
    if s5 != 0 and s10 == 0:
        return "weakened"
    if s5 != 0 and s10 == -s5:
        return "reversed"
    return "resolved"


def _ceiling_minus_s(q: dict, mm: dict, source: str) -> float:
    ok = sorted(q["per_model"])
    if not ok:
        return float("nan")
    return float(np.mean([mm[m][source]["ceiling"] for m in ok]) - np.mean([mm[m][source]["S"] for m in ok]))


def _p_conjunction(*qs: dict) -> float:
    ps = [q["p_one_sided"] for q in qs]
    if any(q["undetermined"] for q in qs) or not all(math.isfinite(x) for x in ps):
        return float("nan")
    return max(ps)


PRIMARY_LABELS = {
    "T5.1": "T5.1 D^P_C1 = AUROC(P) - AUROC(S), trained",
    "D^M1": "T5.2 D^M1_C1 = AUROC(M1) - AUROC(S)",
    "D^M3": "T5.2 D^M3_C1 = AUROC(M3) - AUROC(S)",
    "T5.3": "T5.3 Delta S = AUROC(S, C1) - AUROC(S, C3)",
    "T5.4": "T5.4 D^F2_C1 on stimulus family F2 (valid models)",
    "G^M1": "G^M1 = AUROC(M1) - AUROC(P) (cost of the cheapest proxy)",
    "fraction|T5.1|P": "ceiling fraction of P (B)",
    "fraction|T5.1|S": "ceiling fraction of S (B)",
    "fraction|T5.4|P_F2": "ceiling fraction of P_F2 (F2)",
    "fraction|T5.4|S": "ceiling fraction of S (F2)",
}


def replication(data: dict) -> tuple[dict, dict]:
    """Week 5: quantities per version, decisions and their combination; also returns per-model metrics."""
    st_ = data["structure"]
    size = None if st_ is None else st_["n_krawedzi"].astype(np.float64)
    mm = {m: model_metrics(d, size) for m, d in data["models"].items()}
    mf2 = {m: {"F2": mm[m]["F2"]} for m in mm if "F2" in mm[m]}

    def diff(src, a, b, pool=None):
        pool = mm if pool is None else pool
        return _per_model(pool, lambda x: x[src][a] - x[src][b])

    def fraction(src, method, pool):
        return _per_model(pool, lambda x: M.ceiling_fraction(x[src][method], x[src]["ceiling"]))

    out: dict = {"f2_validity": {m: mm[m]["F2_validity"] for m in sorted(mm) if "F2_validity" in mm[m]},
                 "versions": {}}
    for v, plan in VERSIONS.items():
        Q = {
            "T5.1": quantity(diff("C1", "P", "S"), plan),
            "D^M1": quantity(diff("C1", "M1", "S"), plan),
            "D^M3": quantity(diff("C1", "M3", "S"), plan),
            "T5.3": quantity(_per_model(mm, lambda x: x["C1"]["S"] - x["C3"]["S"]), plan, -1),
            "T5.4": quantity(diff("F2", "P_F2", "S", mf2), plan),
            "G^M1": quantity(diff("C1", "M1", "P"), plan),
            "fraction|T5.1|P": quantity(fraction("C1", "P", mm), plan),
            "fraction|T5.1|S": quantity(fraction("C1", "S", mm), plan),
            "fraction|T5.4|P_F2": quantity(fraction("F2", "P_F2", mf2), plan),
            "fraction|T5.4|S": quantity(fraction("F2", "S", mf2), plan),
        }
        p = {
            "T5.1": Q["T5.1"]["p_one_sided"] if not Q["T5.1"]["undetermined"] else float("nan"),
            "T5.2": _p_conjunction(Q["D^M1"], Q["D^M3"]),
            "T5.3": Q["T5.3"]["p_one_sided"] if not Q["T5.3"]["undetermined"] else float("nan"),
            "T5.4": Q["T5.4"]["p_one_sided"] if not Q["T5.4"]["undetermined"] else float("nan"),
        }
        h = holm(p, alpha=HOLM_ALPHA, m=len(PRIMARY))
        cms = {"T5.1": _ceiling_minus_s(Q["T5.1"], mm, "C1"), "T5.4": _ceiling_minus_s(Q["T5.4"], mf2, "F2")}

        def decisions(hh, Q=Q, cms=cms):
            return {
                "T5.1": decide_t51(Q["T5.1"], cms["T5.1"], hh["T5.1"]),
                "T5.2": decide_t52(Q["D^M1"], Q["D^M3"], hh["T5.2"]),
                "T5.3": decide_t53(Q["T5.3"], hh["T5.3"]),
                "T5.4": decide_t51(Q["T5.4"], cms["T5.4"], hh["T5.4"]),
            }

        out["versions"][v] = {
            "models": list(plan),
            "quantities": [{"key": k, "label": PRIMARY_LABELS[k], **q} for k, q in Q.items()],
            "p_one_sided": p, "holm": h, "ceiling_minus_S": cms,
            "decisions": decisions(h), "decisions_without_holm": decisions(dict.fromkeys(PRIMARY)),
            "n_valid_F2": sum(1 for m in plan if out["f2_validity"].get(m, {}).get("valid")),
        }
    w = out["versions"]
    out["combination"] = {
        k: {"d5": None if k == "T5.4" else w["original"]["decisions_without_holm"][k],
            "d10": w["pooled"]["decisions"][k], "dn": w["replication"]["decisions"][k]}
        for k in PRIMARY
    }
    for c in out["combination"].values():
        c["label"] = combine(c["d5"], c["d10"], c["dn"])
    return out, mm


def method_means(mm: dict) -> list[dict]:
    rows = []
    mf2 = {m: x for m, x in mm.items() if "F2" in x}
    for src, methods, pool in (("C1", ("S", "A", "P", "M1", "M3", "M_akt", "M5", "size", "ceiling"), mm),
                               ("F2", ("S", "A", "P_F2", "M1_F2", "size", "ceiling"), mf2)):
        for met in methods:
            q = quantity(_per_model(pool, lambda x, s=src, me=met: x[s][me]), MODELS_POOLED)
            rows.append({"key": f"{src}|{met}", "label": f"{src} {met}", **q})
    return rows


# ---------------------------------------------------------------- leaderboard (added after the preregistration)

LEADERBOARD = ("P", "M1", "M3", "M_akt", "A", "M5", "S", "size")
TABLE_METRICS = ("AUROC", "signal_share", "AUPRC", "precision_at_61")


def ceilings(data: dict, condition: str = "C1") -> dict[str, float]:
    """AUROC ceiling of the truth per model: E on half B1 against the binary truth from half B2."""
    kind = f"B_{condition}"
    return {m: M.measure("AUROC", d[kind]["E_B1"], d[kind]["E_B2"])
            for m, d in sorted(data["models"].items()) if kind in d}


def summaries(per_model: dict[str, dict[str, float]]) -> dict:
    """{metric: {version: summary}} of per-model leaderboard metrics, for all 10 models and both halves."""
    return {metric: {v: summarize(_per_model(per_model, lambda x, mi=metric: x[mi]), plan).to_dict()
                     for v, plan in VERSIONS.items()} for metric in TABLE_METRICS}


def leaderboard(data: dict) -> dict:
    """Every method in the trained and untrained networks on the four table metrics.

    AUROC is the preregistered primary metric. The share of the achievable signal, (AUROC - 0.5) / (ceiling -
    0.5) per model, AUPRC and precision@61 are secondary and were added after the preregistration
    (`docs/preregistration.md`, deviations); they reframe the same rankings and change no decision.
    """
    st_ = data["structure"]
    size = None if st_ is None else st_["n_krawedzi"].astype(np.float64)
    per: dict = {}
    for cond in CONDITIONS:
        ceil = ceilings(data, cond)
        kind = f"B_{cond}"
        for m in sorted(ceil):
            d = data["models"][m]
            b = d[kind]
            scores = c1_scores(d, size) if cond == "C1" else {k: b[k] for k in ("S", "A", "P")}
            row = {k: M.method_metrics(v, b["E_B"], ceil[m]) for k, v in scores.items()}
            row["ceiling"] = M.method_metrics(b["E_B1"], b["E_B2"], ceil[m])
            per.setdefault(cond, {})[m] = row
    rows = []
    for cond, methods in (("C1", (*LEADERBOARD, "ceiling")), ("C3", ("S", "A", "P", "ceiling"))):
        for met in methods:
            pm = {m: x[met] for m, x in per.get(cond, {}).items() if met in x}
            rows.append({"key": f"{cond}|{met}", "condition": cond, "method": met, **summaries(pm)})
    return {"rows": rows, "per_model": per}


README_EXAMPLE = "mean |weight| per edge"


def readme_example(data: dict) -> dict | None:
    """The method of the README example ("Add your method"): sum of |weights| / number of edges of each class,
    trained networks, scored exactly as `fcb score` does. Quoted in the README; not a preregistered method."""
    if data["structure"] is None:
        return None
    n_edges = data["structure"]["n_krawedzi"].astype(np.float64)
    ceil = ceilings(data, "C1")
    per = {m: M.method_metrics(np.asarray(data["models"][m]["B_C1"]["A"], np.float64) / n_edges,
                               data["models"][m]["B_C1"]["E_B"], ceil[m]) for m in sorted(ceil)}
    return {"name": README_EXAMPLE, **summaries(per)}


# ---------------------------------------------------------------- controls K1-K5


def type_graph(sources, targets) -> dict:
    """Out/in degree of cell types and depth (BFS from photoreceptors R1-R8) in the type-to-type graph."""
    src = [str(z) for z in sources]
    tgt = [str(c) for c in targets]
    types = sorted(set(src) | set(tgt))
    outs: dict[str, set] = {t: set() for t in types}
    ins: dict[str, set] = {t: set() for t in types}
    for z, c in zip(src, tgt, strict=True):
        outs[z].add(c)
        ins[c].add(z)
    depth: dict[str, int] = {t: 0 for t in PHOTORECEPTORS if t in outs}
    queue = deque(sorted(depth))
    while queue:
        t = queue.popleft()
        for c in sorted(outs[t]):
            if c not in depth:
                depth[c] = depth[t] + 1
                queue.append(c)
    deepest = max(depth.values()) if depth else 0
    for t in types:
        depth.setdefault(t, deepest + 1)
    return {"out": {t: len(outs[t]) for t in types}, "in": {t: len(ins[t]) for t in types}, "depth": depth}


def trivial_baselines(sources, targets, n_edges) -> dict[str, np.ndarray]:
    """Scores of trivial baselines per class; direction fixed in advance (larger = more important)."""
    g = type_graph(sources, targets)
    src = [str(z) for z in sources]
    tgt = [str(c) for c in targets]
    return {
        "class size (edges)": np.asarray(n_edges, dtype=np.float64),
        "source out-degree": np.array([g["out"][z] for z in src], dtype=np.float64),
        "source in-degree": np.array([g["in"][z] for z in src], dtype=np.float64),
        "target out-degree": np.array([g["out"][c] for c in tgt], dtype=np.float64),
        "target in-degree": np.array([g["in"][c] for c in tgt], dtype=np.float64),
        "minus source depth": -np.array([g["depth"][z] for z in src], dtype=np.float64),
        "minus target depth": -np.array([g["depth"][c] for c in tgt], dtype=np.float64),
    }


def controls(data: dict) -> dict:
    models = data["models"]
    d = {c: {m: models[m][f"B_{c}"] for m in MODELS_ORIGINAL} for c in CONDITIONS}
    first = MODELS_ORIGINAL[0]
    s = d["C1"][first]["S"]
    n_edges = data["structure"]["n_krawedzi"].astype(np.float64)

    def summ(f) -> dict:
        return summarize({m: f(m) for m in MODELS_ORIGINAL}).to_dict()

    a3 = M.quantize(d["C3"][first]["A"])
    k1 = {
        "spearman_S_absw_C3": M.spearman(s, a3),
        "spearman_S_n_edges": M.spearman(s, n_edges),
        "spearman_S_absw_C1": summ(lambda m: M.spearman(s, d["C1"][m]["A"])),
        "C3_sum_absw_equals_0.01_x_edges_max_rel_diff": float(
            M.relative_difference(d["C3"][first]["A"], C3_SCALE * n_edges).max()),
    }
    k2 = {}
    for c in CONDITIONS:
        dc = d[c]

        def truth(m, key, dc=dc):
            return M.binary_truth(dc[m][key])

        ceil = {m: M.auroc(dc[m]["E_B1"], truth(m, "E_B2")) for m in MODELS_ORIGINAL}
        k2[c] = {
            "auroc_E_B1_vs_truth_B2": summarize(ceil).to_dict(),
            "spearman_E_B1_E_B2": summ(lambda m, dc=dc: M.spearman(dc[m]["E_B1"], dc[m]["E_B2"])),
            "jaccard_top10_B1_B2": summ(lambda m, truth=truth: M.jaccard(truth(m, "E_B1"), truth(m, "E_B2"))),
            "ceiling_fraction_P": summ(lambda m, dc=dc, truth=truth, ceil=ceil: M.ceiling_fraction(
                M.auroc(dc[m]["P"], truth(m, "E_B")), ceil[m])),
            "ceiling_fraction_S": summ(lambda m, dc=dc, truth=truth, ceil=ceil: M.ceiling_fraction(
                M.auroc(dc[m]["S"], truth(m, "E_B")), ceil[m])),
        }
    base = trivial_baselines(d["C1"][first]["zrodlo"], d["C1"][first]["cel"], n_edges)
    k3 = {}
    for c in CONDITIONS:
        dc = d[c]
        k3[c] = {name: summ(lambda m, x=x, dc=dc: M.auroc(x, M.binary_truth(dc[m]["E_B"])))
                 for name, x in base.items()}
        k3[c]["S"] = summ(lambda m, dc=dc: M.auroc(dc[m]["S"], M.binary_truth(dc[m]["E_B"])))
    k4 = {}
    for c in CONDITIONS:
        dc = d[c]

        def abs_truth(m, dc=dc):
            return M.binary_truth(M.absolute_effect(dc[m]["wzgledne"], dc[m]["r_ref_norma"]))

        k4[c] = {
            "jaccard_top10_relative_absolute": summ(
                lambda m, dc=dc, abs_truth=abs_truth: M.jaccard(M.binary_truth(dc[m]["E_B"]), abs_truth(m))),
            "D_P_S_absolute_truth": summ(lambda m, dc=dc, abs_truth=abs_truth: M.auroc(dc[m]["P"], abs_truth(m))
                                         - M.auroc(dc[m]["S"], abs_truth(m))),
            "r_norm_range": [float(min(dc[m]["r_ref_norma"].min() for m in MODELS_ORIGINAL)),
                             float(max(dc[m]["r_ref_norma"].max() for m in MODELS_ORIGINAL))],
        }
    k5 = None
    if data.get("k5") is not None:
        j = data["k5"]
        classes = [int(i) for i in j["klasy"]]
        stored = np.asarray(j["E_zapisane"], dtype=np.float64)
        other = np.asarray(j["E_druga_sciezka"], dtype=np.float64)
        package = np.asarray(d["C1"][first]["E_B"], dtype=np.float64)[classes]
        k5 = {"model": first, "condition": "C1", "n_classes": len(classes),
              "max_rel_diff_independent_path": float(M.relative_difference(stored, other).max()),
              "max_rel_diff_package_vs_stored": float(M.relative_difference(package, stored).max()),
              "threshold": K5_THRESHOLD}
        k5["passed"] = bool(k5["max_rel_diff_independent_path"] < K5_THRESHOLD
                            and k5["max_rel_diff_package_vs_stored"] < K5_THRESHOLD)
    return {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}


# ---------------------------------------------------------------- everything


def headline(w1: dict, rep: dict) -> dict:
    t51 = next(q for q in rep["versions"]["pooled"]["quantities"] if q["key"] == "T5.1")
    return {"D_C1_N5": w1["failure_criterion"]["D_C1"], "D_C1_N10": {k: t51[k] for k in (
        "mean", "ci_bootstrap", "ci_t", "per_model", "min", "max", "n", "sign")},
        "W1_failed": w1["failure_criterion"]["failed"]}


def analyze(data: dict) -> dict:
    w1 = week1(data)
    rep, mm = replication(data)
    return {
        "headline": headline(w1, rep),
        "week1": {k: v for k, v in w1.items() if k != "per_model"},
        "week1_per_model": w1["per_model"],
        "mechanism": mechanism(w1),
        "replication": rep,
        "method_means": method_means(mm),
        "leaderboard": leaderboard(data),
        "readme_example": readme_example(data),
        "per_model_auroc": {m: {k: v for k, v in x.items() if k in ("C1", "C3", "F2")} for m, x in mm.items()},
        "controls": controls(data),
    }
