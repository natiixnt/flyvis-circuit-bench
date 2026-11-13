"""Rendering of the analysis: `results/results.json` (machine readable) and `docs/results.md` (full tables).

Both outputs are deterministic: JSON floats are rounded to 12 significant digits (absorbing last-bit differences
between BLAS builds), markdown shows 3 decimals. `fcb reproduce --check` compares both byte for byte.
"""

from __future__ import annotations

import json
import math

from .names import CONDITIONS as COND
from .names import COST, DAGGER_NOTE, DECISIONS, REFUTED_SENTENCE
from .names import METHODS as NAME


def _clean(x):
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float(f"{x:.12g}")
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        return _clean(x.item())
    return x


def to_json(results: dict) -> str:
    return json.dumps(_clean(results), indent=1, sort_keys=False, ensure_ascii=True) + "\n"


def f(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.3f}"


def fp(p) -> str:
    if isinstance(p, float) and math.isfinite(p) and 0 < p < 1e-3:
        return f"{p:.1e}"
    return f(p)


def ci(pair) -> str:
    return f"[{f(pair[0])}, {f(pair[1])}]"


def models(per_model: dict) -> str:
    return "; ".join(f"{m} {f(v)}" for m, v in sorted(per_model.items()))


HEAD = ["| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |",
        "|---|---|---|---|---|---|---|---|"]


def cell(text: str) -> str:
    """Table cell text: a literal | (as in |weights|) is escaped so it does not split the cell."""
    return text.replace("|", "\\|")


def row(label: str, q: dict) -> str:
    sign = q["sign"]
    if q.get("no_effect_above_mde"):
        sign += " (no effect > MDE)"
    if q.get("omitted"):
        sign += f" (without {', '.join(q['omitted'])})"
    return (f"| {cell(label)} | {f(q['mean'])} | {ci(q['ci_bootstrap'])} | {ci(q['ci_t'])} | {models(q['per_model'])} "
            f"| {f(q['min'])} | {f(q['max'])} | {sign} |")


INTRO = """# Results

All numbers below are produced by `fcb reproduce` from the measurement package and are identical to the tables
of the original study. Notation: mean over models, 95% CI from the percentile bootstrap over models (10 000
resamples, seed 20261101) and from Student t (df = N - 1). Sign rule: "positive" only when the lower bounds of
both intervals are > 0, "negative" only when both upper bounds are < 0, otherwise "inconclusive". Ground truth:
top 10% of the 604 connection classes by the ablation effect E (61 classes per model and network).

Networks: trained = the released flyvis models; untrained = the same networks at flyvis initialization.
Discovery = models 000-004 (analysed first), replication = models 005-009 (analysed after the replication rules
were fixed). Methods: synapse count of the class (anatomy), sum of |weights| of the class, attribution patching
(32 random output directions), gradient (one direction), |weights| x neuron activity (per edge: |weight| x mean
activity of the individual source neuron on held-out stimuli, summed over the class), |weights| x source activity
(sum of |weights| of the class x mean |activity| of the source cell type), |weights| x source SAE code (sum of
|weights| x mean L1 norm of the sparse autoencoder code of the source type), source activity only (mean |activity|
of the source type, no weights), synapse count x source activity (no learned weights), class size (number of
neuron-to-neuron edges). "truth ceiling" = reliability of the truth itself: AUROC of E on one half of the
stimuli against the truth from the other half. "top-61 Spearman" = Spearman with E
inside the 61 positive classes, the one measure that does not saturate in untrained networks.
The short codes of the measurement files are listed in [methodology.md](methodology.md#glossary).
"""

MEASURE = {"AUROC": "AUROC", "AUPRC": "AUPRC", "Spearman": "Spearman", "c1": "top-61 Spearman"}

CONTRAST = {
    "T5.1": "patching minus synapse count",
    "D^M1": "gradient minus synapse count",
    "D^M3": "|weights| x neuron activity minus synapse count",
    "T5.3": "synapse count, trained minus untrained",
    "T5.4": "patching minus synapse count, second stimulus set",
    "G^M1": "gradient minus patching (cost of one backward pass)",
    "fraction|T5.1|P": "share of signal, patching",
    "fraction|T5.1|S": "share of signal, synapse count",
    "fraction|T5.4|P_F2": "share of signal, patching, second stimulus set",
    "fraction|T5.4|S": "share of signal, synapse count, second stimulus set",
}


def contrast_label(key: str) -> str:
    """Plain label of a discovery contrast key: `C1|P-S|AUROC` or `DD|P-S|AUROC`."""
    c, ab, mi = key.split("|")
    a, b = ab.split("-")
    if c == "DD":
        return f"{mi}: (patching minus {NAME[b]}) trained minus untrained"
    return f"{mi}: {NAME[a]} minus {NAME[b]}, {COND[c]}"


def mechanism_label(key: str) -> str:
    kind, mi = key.split("|")
    if kind == "dS":
        return f"{MEASURE[mi]}: synapse count, trained minus untrained"
    return f"{MEASURE[mi]}: (patching minus synapse count) trained minus untrained"


def mean_label(key: str) -> str:
    src, met = key.split("|")
    return f"{COND[src]}: {NAME[met]}"


def render_markdown(r: dict) -> str:
    L = [INTRO]
    w1 = r["week1"]
    L += ["## 1. Discovery: methods against the ablation truth (models 000-004)", "",
          "Mean over models with the 95% bootstrap CI.", "",
          "| network | method | AUROC | AUPRC | Spearman |", "|---|---|---|---|---|"]
    by = {x["key"]: x for x in w1["methods"]}
    for c in ("C1", "C3"):
        for method in ("S", "A", "P", "ceiling"):
            cells = [f"{f(by[f'{c}|{method}|{mi}']['mean'])} {ci(by[f'{c}|{method}|{mi}']['ci_bootstrap'])}"
                     for mi in ("AUROC", "AUPRC", "Spearman")]
            L.append(f"| {COND[c]} | {cell(NAME[method])} | {' | '.join(cells)} |")
    L += ["", "### Preregistered contrasts", "", *HEAD]
    L += [row(contrast_label(x["key"]), x) for x in w1["contrasts"]]
    fc = w1["failure_criterion"]
    L += ["", "### Preregistered failure criterion (trained networks)", "",
          f"- (1) patching minus synapse count = {f(fc['D_C1']['mean'])}, bootstrap {ci(fc['D_C1']['ci_bootstrap'])}, "
          f"t {ci(fc['D_C1']['ci_t'])}; a lower bound <= 0: **{f(fc['condition_1_lower_bound_not_positive'])}**.",
          f"- (2) AUROC of synapse count = {f(fc['auroc_S'])}, ceiling = {f(fc['auroc_ceiling'])}, distance "
          f"{f(fc['ceiling_minus_S'])} <= 0.05: **{f(fc['condition_2_S_near_ceiling'])}**.",
          f"- Decision: **the benchmark {'fails' if fc['failed'] else 'survives'}**.", ""]
    L += ["## 2. Follow-up: is the advantage of patching a mechanism of training? (models 000-004)", "",
          "In untrained networks the AUROC of patching saturates at 1.000, so the difference of gaps between "
          "trained and untrained networks on AUROC is bounded by the ceiling. The preregistered follow-up uses "
          "the top-61 Spearman, the only measure that does not saturate there. On it the sign flips. "
          f"{REFUTED_SENTENCE}", "", *HEAD]
    L += [row(mechanism_label(x["key"]), x) for x in r["mechanism"]]
    rep = r["replication"]
    L += ["", "## 3. Replication with 10 models and a second stimulus set", "",
          "Four primary decisions fixed before any number of models 005-009 or of the second stimulus set "
          "existed. Holm correction within each version, one-sided level 0.025, m = 4. Discovery = rules of the "
          "first analysis on 000-004 (no Holm), all 10 = pooled 000-009 (main estimate), replication = 005-009.",
          "", "| decision | discovery | all 10 | replication | combination | p all 10 | Holm all 10 | "
          "p replication | Holm replication |", "|---|---|---|---|---|---|---|---|---|"]
    vp, vn = rep["versions"]["pooled"], rep["versions"]["replication"]
    for k, c in rep["combination"].items():
        L.append(f"| {DECISIONS[k]} | {c['d5'] or 'n/a (set added later)'} | {c['d10']} | {c['dn']} | "
                 f"{c['label']} | {fp(vp['p_one_sided'][k])} | {f(vp['holm'][k])} | {fp(vn['p_one_sided'][k])} | "
                 f"{f(vn['holm'][k])} |")
    L += ["", "Validity of the second stimulus set (conditions fixed before any of its effects existed): (i) >= 50% "
          "of classes with E > 0, (ii) q90/median >= 3 and finite max, (iii) ceiling AUROC >= 0.95, (iv) >= 20 of "
          "24 bars with nonzero response.", "",
          "| model | (i) classes E > 0 | (ii) q90/median | (iii) ceiling | (iv) bars | valid |",
          "|---|---|---|---|---|---|"]
    for m, v in rep["f2_validity"].items():
        L.append(f"| {m} | {v['i_classes_E_positive']} | {f(v['ii_q90_over_median'])} | "
                 f"{f(v['iii_ceiling_auroc'])} | {v['iv_nonzero_bars']} | {f(v['valid'])} |")
    for v, title in (("pooled", "all 10 models (000-009)"), ("original", "discovery (000-004)"),
                     ("replication", "replication (005-009)")):
        ver = rep["versions"][v]
        dec = "; ".join(f"{DECISIONS[k]}: {d}" for k, d in ver["decisions"].items())
        L += ["", f"### Version: {title}", "", f"Decisions: {dec}. Ceiling minus AUROC of synapse count: "
              f"{f(ver['ceiling_minus_S']['T5.1'])}, on the second stimulus set "
              f"{f(ver['ceiling_minus_S']['T5.4'])} (threshold > 0.05). Models valid on the second stimulus set: "
              f"{ver['n_valid_F2']} of {len(ver['models'])}.", "", *HEAD]
        L += [row(CONTRAST[q["key"]], q) for q in ver["quantities"]]
    L += ["", "## 4. Mean AUROC of every method (descriptive, 10 models)", "", *HEAD]
    L += [row(mean_label(x["key"]), x) for x in r["method_means"]]
    L += render_controls(r["controls"])
    L += render_leaderboard(r["leaderboard"])
    if r.get("readme_example"):
        L += render_example(r["readme_example"])
    return "\n".join(L).rstrip() + "\n"


def render_controls(k: dict) -> list[str]:
    L = ["", "## 5. Validity controls (models 000-004)", "",
         "Run before the replication data existed, to rule out that the result comes from a shared bug or a "
         "trivial explanation (truth and methods share models and code).", ""]
    k1 = k["K1"]
    L += ["### Control 1: the untrained initialization carries synapse-count structure by construction", "",
          f"- Spearman(synapse count, sum of |w| per class, untrained) = {f(k1['spearman_S_absw_C3'])} (identical "
          f"across models); Spearman(synapse count, edges per class) = {f(k1['spearman_S_n_edges'])}.",
          f"- Spearman(synapse count, sum of |w|, trained) = {f(k1['spearman_S_absw_C1']['mean'])} "
          f"{ci(k1['spearman_S_absw_C1']['ci_bootstrap'])} | t {ci(k1['spearman_S_absw_C1']['ci_t'])}.",
          f"- Untrained, the sum of |w| of a class equals 0.01 x number of edges (max relative difference "
          f"{f(k1['C3_sum_absw_equals_0.01_x_edges_max_rel_diff'])}), so the untrained network is \"initialized "
          "from the connectome\", not \"unstructured\".", ""]
    L += ["### Control 2: noise ceiling of the truth", "", *HEAD]
    for c in ("C1", "C3"):
        for key, label in (("auroc_E_B1_vs_truth_B2", "AUROC(E on half 1, truth from half 2)"),
                           ("spearman_E_B1_E_B2", "Spearman(E on half 1, E on half 2)"),
                           ("jaccard_top10_B1_B2", "Jaccard of top 10% on the two halves"),
                           ("ceiling_fraction_P", "share of signal, patching"),
                           ("ceiling_fraction_S", "share of signal, synapse count")):
            L.append(row(f"{COND[c]}: {label}", k["K2"][c][key]))
    L += ["", "### Control 3: trivial baselines (AUROC)", "", *HEAD]
    for c in ("C1", "C3"):
        for name, q in k["K3"][c].items():
            L.append(row(f"{COND[c]}: {NAME.get(name, name)}", q))
    L += ["", "### Control 4: normalization of E", "", *HEAD]
    for c in ("C1", "C3"):
        L.append(row(f"{COND[c]}: Jaccard top 10% (relative E vs absolute E)",
                     k["K4"][c]["jaccard_top10_relative_absolute"]))
        L.append(row(f"{COND[c]}: patching minus synapse count (AUROC) on the absolute truth",
                     k["K4"][c]["D_P_S_absolute_truth"]))
    L.append("")
    k5 = k["K5"]
    if k5 is not None:
        L += ["### Control 5: independent ablation", "",
              f"- {k5['n_classes']} classes of model {k5['model']}, trained (10 large, 10 medium, 10 small "
              "effects), re-ablated by a separate code path (zeroing synapse counts, public flyvis `simulate`, "
              "central cells from hex coordinates, separate formula for E).",
              f"- Max relative difference of E between the two paths: {f(k5['max_rel_diff_independent_path'])}; "
              f"between the package file and the stored control: {f(k5['max_rel_diff_package_vs_stored'])} "
              f"(threshold {k5['threshold']:g}). Passed: **{f(k5['passed'])}**.", ""]
    return L


def pct(x) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{100 * x:.0f}%"


LEADERBOARD_HEAD = ["| method | needs | AUROC [95% CI] | share of signal | discovery | replication | AUPRC | "
                    "precision@61 |", "|---|---|---|---|---|---|---|---|"]


def _lb_row(x: dict) -> str:
    a = x["AUROC"]["pooled"]
    return (f"| {cell(NAME[x['method']])} | {COST[x['method']]} | {f(a['mean'])} {ci(a['ci_bootstrap'])} | "
            f"{pct(x['signal_share']['pooled']['mean'])} | {f(x['AUROC']['original']['mean'])} | "
            f"{f(x['AUROC']['replication']['mean'])} | {f(x['AUPRC']['pooled']['mean'])} | "
            f"{f(x['precision_at_61']['pooled']['mean'])} |")


def leaderboard_table(r: dict) -> str:
    """README leaderboard: trained networks, 10 models, methods by AUROC, the truth ceiling last."""
    rows = [x for x in r["leaderboard"]["rows"] if x["condition"] == "C1"]
    methods = sorted((x for x in rows if x["method"] != "ceiling"), key=lambda x: -x["AUROC"]["pooled"]["mean"])
    ceiling = [x for x in rows if x["method"] == "ceiling"]
    return "\n".join([*LEADERBOARD_HEAD, *(_lb_row(x) for x in methods + ceiling)]) + "\n"


def main_table(r: dict) -> str:
    """Summary printed by `fcb reproduce` (not in the README): AUROC of the four reference rows, all 10
    models and per half, and share of signal."""
    by = {x["key"]: x for x in r["leaderboard"]["rows"]}
    L = ["| method | trained, all 10 | discovery (000-004) | replication (005-009) | share of signal | "
         "untrained, all 10 |", "|---|---|---|---|---|---|"]
    for met in ("S", "A", "P", "ceiling"):
        t, u = by[f"C1|{met}"], by[f"C3|{met}"]
        L.append(f"| {cell(NAME[met])} | {f(t['AUROC']['pooled']['mean'])} | {f(t['AUROC']['original']['mean'])} | "
                 f"{f(t['AUROC']['replication']['mean'])} | {pct(t['signal_share']['pooled']['mean'])} | "
                 f"{f(u['AUROC']['pooled']['mean'])} |")
    return "\n".join(L) + "\n"


def render_leaderboard(lb: dict) -> list[str]:
    L = ["## 6. Leaderboard: every method on four metrics (10 models)", "",
         "AUROC is the preregistered primary metric. Share of signal = (AUROC - 0.5) / (ceiling - 0.5) per model, "
         "AUPRC and precision@61 (how many of the 61 most important classes are in the method's top 61) are "
         "secondary metrics added after the preregistration; they change no decision. " + DAGGER_NOTE, ""]
    for c in ("C1", "C3"):
        for metric, label in (("AUROC", "AUROC"), ("signal_share", "share of signal"), ("AUPRC", "AUPRC"),
                              ("precision_at_61", "precision@61")):
            L += [f"### {COND[c].capitalize()} networks: {label}", "", *HEAD]
            for x in lb["rows"]:
                if x["condition"] != c:
                    continue
                for v, vname in (("pooled", "all 10"), ("original", "discovery"), ("replication", "replication")):
                    L.append(row(f"{NAME[x['method']]}, {vname}", x[metric][v]))
            L.append("")
    return L


def render_example(x: dict) -> list[str]:
    L = ["### The README example: " + cell(x["name"]) + " (trained networks)", "",
         "The example method of the README section \"Add your method\" (sum of |weights| / number of edges of "
         "each class), scored as `fcb score` does. Not a preregistered method.", "", *HEAD]
    for metric, label in (("AUROC", "AUROC"), ("signal_share", "share of signal"), ("AUPRC", "AUPRC"),
                          ("precision_at_61", "precision@61")):
        L.append(row(f"{label}, all 10", x[metric]["pooled"]))
    return [*L, ""]


TOP_N = 61  # size of the binary truth: the top 10% of the 604 classes


def readme_facts(r: dict) -> dict[str, str]:
    """The numbers quoted in the README text (TL;DR, captions), formatted as they appear there."""
    lb = {x["key"]: x for x in r["leaderboard"]["rows"]}
    mm = {x["key"]: x for x in r["method_means"]}
    n10 = r["headline"]["D_C1_N10"]
    dd = next(q for q in r["mechanism"] if q["key"] == "DD|c1")
    dda = next(q for q in r["mechanism"] if q["key"] == "DD|AUROC")
    g = next(q for q in r["replication"]["versions"]["pooled"]["quantities"] if q["key"] == "G^M1")
    out = {}
    for met in ("S", "A", "P", "M1", "M3", "M_akt", "M5", "act", "S_act", "size", "ceiling"):
        out[f"auroc_{met}"] = f(mm[f"C1|{met}"]["mean"])
        out[f"share_{met}"] = pct(lb[f"C1|{met}"]["signal_share"]["pooled"]["mean"])
        out[f"p61_{met}"] = f(lb[f"C1|{met}"]["precision_at_61"]["pooled"]["mean"])
    out["auroc_untrained_S"] = f(lb["C3|S"]["AUROC"]["pooled"]["mean"])
    out["auroc_untrained_P"] = f(lb["C3|P"]["AUROC"]["pooled"]["mean"])
    out["gap"] = f(n10["mean"])
    out["gap_ci"] = ci(n10["ci_bootstrap"])
    out["gap_ci_t"] = ci(n10["ci_t"])
    out["gap_positive"] = f"{sum(v > 0 for v in n10['per_model'].values())} of {len(n10['per_model'])}"
    out["dd_top61"] = f(dd["mean"])
    out["dd_top61_ci"] = ci(dd["ci_bootstrap"])
    out["dd_auroc"] = f(dda["mean"])
    out["gradient_cost"] = f(g["mean"])
    out["gradient_loss"] = f(-g["mean"])
    out["gradient_cost_ci"] = ci(g["ci_bootstrap"])
    out["gradient_cost_ci_t"] = ci(g["ci_t"])
    n5 = r["headline"]["D_C1_N5"]
    out["gap_discovery"] = f(n5["mean"])
    out["gap_discovery_ci"] = ci(n5["ci_bootstrap"])
    out["gap_discovery_ci_t"] = ci(n5["ci_t"])
    out["dd_top61_ci_t"] = ci(dd["ci_t"])
    out["dd_auroc_ci"] = ci(dda["ci_bootstrap"])
    out["dd_auroc_ci_t"] = ci(dda["ci_t"])
    out["spearman_untrained_weights"] = f(r["controls"]["K1"]["spearman_S_absw_C3"])
    for met in ("S", "A", "P", "M1", "M3", "M_akt", "M5", "act", "S_act", "size", "ceiling"):
        a = lb[f"C1|{met}"]["AUROC"]
        out[f"auroc_ci_{met}"] = ci(a["pooled"]["ci_bootstrap"])
    # precision@61 as a count: how many of the 61 most important classes synapse count puts in its top 61
    out["p61_count_S"] = str(round(TOP_N * lb["C1|S"]["precision_at_61"]["pooled"]["mean"]))
    # the part of the gap left by synapse count that one learned scalar per class (sum of |weights|) closes
    s_s, s_a = (lb[f"C1|{k}"]["signal_share"]["pooled"]["mean"] for k in ("S", "A"))
    out["gap_closed_A"] = pct((s_a - s_s) / (1 - s_s))
    ex = r.get("readme_example")
    if ex:
        out["example_auroc"] = f(ex["AUROC"]["pooled"]["mean"])
    if r.get("v1_1"):
        out.update(readme_facts_v11(r["v1_1"]))
    return out


def readme_facts_v11(v: dict) -> dict[str, str]:
    """The numbers of the README section on the v1.1 analyses (and the one TL;DR sentence), as printed there."""
    out: dict[str, str] = {}
    if "z1" in v:
        z = v["z1"]
        for met in ("S", "P"):
            a = [x["AUROC"]["pooled"]["mean"] for x in z["table"] if x["method"] == met]
            out[f"v11_z1_{met}_lo"], out[f"v11_z1_{met}_hi"] = f(min(a)), f(max(a))
        out["v11_z1_swaps"] = str(len(z["pairs"]["pooled"]["swaps"]))
        out["v11_z1_pairs"] = str(len(z["methods"]) * (len(z["methods"]) - 1) // 2)
    if "z2" in v:
        z = v["z2"]
        out["v11_z2_spearman"] = f(z["agreement"]["spearman"]["pooled"]["mean"])
        for fam in ("edges", "sintel"):
            out[f"v11_z2_P_{fam}"] = f(z["ladder"]["P"][fam]["AUROC"]["pooled"]["mean"])
            out[f"v11_z2_S_{fam}"] = f(z["ladder"]["S"][fam]["AUROC"]["pooled"]["mean"])
    if "z3" in v:
        z = v["z3"]
        out["v11_z3_median_spearman"] = f"{z['pairs']['pooled']['spearman']['median']:.2f}"
        out["v11_z3_overlap"] = f"{z['pairs']['pooled']['overlap']['median']:.0f}"
        out["v11_z3_core"] = str(z["blocks"]["pooled"]["group_sizes"]["core"])
        out["v11_z3_core_min"] = str(z["blocks"]["pooled"]["core_min"])
        out["v11_z3_split_half"] = f"{z['split_half_spearman']['pooled']['mean']:.2f}"
        c = z["cross_model_auroc"]
        out["v11_z3_other_model"] = f(c["model_a_e"]["mean"])
        out["v11_z3_anatomy"] = f(c["synapse_count"]["mean"])
        out["v11_z3_better"], out["v11_z3_pairs"] = str(c["n_pairs_a_better"]), str(c["n_pairs"])
    if "z4" in v:
        z = v["z4"]
        out["v11_z4_spearman"] = f(z["spearman_E_T"]["pooled"]["mean"])
        out["v11_z4_high_low"] = str(z["pooled_mean_table"]["high_E_low_T"])
        out["v11_z4_both"] = str(z["pooled_mean_table"]["high_E_high_T"])
        out["v11_z4_ceiling"] = f(z["ceiling"]["pooled"]["mean"])
    if "z6" in v:
        z = v["z6"]
        out["v11_z6_spearman"] = f(z["spearman_sum"]["pooled"]["mean"])
        out["v11_z6_masking"] = str(len(z["masking_types"]))
        out["v11_z6_reverse"] = str(len(z["reverse_types"]))
        out["v11_z6_exploding"], out["v11_z6_pairs"] = str(z["exploding"]["pairs"]), str(z["exploding"]["pairs_total"])
        out["v11_z6_types"] = str(z["n_types"])
    return out
