# Preregistration (English translation)

## What "preregistered" means here, and what it does not

- The decision rules were written down, in Polish, **before the data they govern existed**: the core plan on
  2026-11-01 before any measurement, one annex per later stage before the first number of that stage, and the
  replication annex on 2026-11-09 before any number of models 005-009 or of the second stimulus set.
- Those documents live in a private working repository. Its history (commits with dates) is **not public** and
  nothing on this page can prove the dates to an outside reader. What we can offer is a commitment: below are the
  sha256 sums of the original Polish document at four points. If we publish the original later, anyone can check
  that it is the same text.
- There is **no external time stamp yet** (OSF, AsPredicted or similar). An external time stamp (OSF) for the
  rules of the next study, before its data exist, is planned; no date is set.
- Every change after a rule was written is listed in [Deviations](#deviations), with its date, its reason and
  whether data already existed.

| version of the original document | date | sha256 |
|---|---|---|
| core plan, before any measurement | 2026-11-01 | `sha256:48a20045f57815ff6df7ef1b0440cc058301e4606b138c71adef5f35336cfc59` |
| with the interval rule, before the first trained-network result | 2026-11-02 | `sha256:e8fd23fc86fa6b0592b4d1f04a08ade7882e7908bd43df26c95a206a914f4251` |
| with the replication annex, before any replication number | 2026-11-09 | `sha256:549ab65feedc02d2090027472ac4d1306af4bc57802199d3758a952615193bc3` |
| final, with the device deviation | 2026-11-10 | `sha256:e6e0142a0a6b517cb7509eb4b01930d2939dd8a91fcd81852fe7302a5d6232ac` |

This page translates the parts that govern the published results: the core plan, the follow-up on the ceiling,
the replication annex and every deviation. The annexes on the cheaper methods are summarized in
[methodology.md](methodology.md). The translation is faithful in content; short codes of the original are
replaced by plain names (the [glossary](methodology.md#glossary) maps them) and file paths of the original
project by the names used in this repository.

Constants for the whole study: main seed 20261101; random numbers only from `numpy.random.default_rng(seed)` (or a
torch generator with that seed); results independent of file order (sorted); 95% CI from the percentile
bootstrap, 10 000 resamples, unless stated otherwise; every comparison is reported with numbers, also when the
result is null.

## Core plan: mask or mechanism (flyvis, no training) - 2026-11-01

**Unit.** A type-to-type connection class in the flyvis network (all classes, not a sample; 604 classes).
Ablation of class e: synaptic scale of that class set to 0, everything else unchanged.

**Stimuli.** A fixed set B of stimuli from the flyvis package (moving edges in all directions and Sintel clips
from its loader), number and list frozen in a file before the first ablation. Split into halves B1 and B2 at
random, with a seed, frozen in the same file.

**Output and effect.** Response r(b): activity of the central column of every cell type over time (all types,
not only those used by the task decoder). Ablation effect E(e) = mean over b of ||r_e(b) - r(b)||² / ||r(b)||²,
computed separately on B, B1 and B2.

**Truth.** Continuous: E(e). Binary: e is "important" if E(e) is in the top 10% of classes in that model and
network.

**Networks (truth computed separately in each).**
- Trained: models of the released flyvis ensemble.
- Shuffled: synaptic scales of the trained model permuted at random between classes (signs and synapse counts
  unchanged). See Deviations: undetermined.
- Untrained: parameters from the flyvis initialization (network before training), same connectome.

**Methods (ranking of classes, without the truth).**
- Synapse count, the anatomical baseline: total synapse count of the class (no model parameters).
- Sum of |weights|, the parameter baseline: synapse count times scale, summed over the class (no intervention).
- Attribution patching: first-order approximation of the same effect. For K = 32 random vectors
  v_k ~ N(0, I) on the output we compute the gradient of v_k . r with respect to the scales s_e;
  score(e) = mean over k and b of (s_e x d(v_k . r)/d s_e)² divided by ||r(b)||². This is an unbiased estimator of
  ||J_e s_e||², the squared linear change of the output, and costs K backward passes instead of one ablation per
  class. Fixed seed for the vectors v.

**Metrics.** AUROC and AUPRC of each method against the binary truth, Spearman against the continuous truth.
Ceiling: reliability of the truth, the AUROC of the ranking by E on B1 against the binary truth on B2 (and
Spearman of E(B1) with E(B2)).

**Number of models.** 5 models of the ensemble (the first five by identifier). If one full run of one model
in one network takes less than 30 min on a laptop CPU, 10. The choice is recorded before measuring the trained
networks.

**Main quantity.** The gap = AUROC(patching) - AUROC(synapse count) in a network, mean over models; 95% CI from
the bootstrap over models. With 10 models or fewer the bootstrap is inaccurate, so we also report every model,
minimum, maximum and the Student t CI. Secondary: the same with sum of |weights| instead of synapse count, AUPRC
instead of AUROC, and the difference between the gap in trained and in untrained networks.

**Failure criterion (kills the hypothesis in this form).** Either of the two, in trained networks:
1. the CI of the gap includes 0 (ranking by synapses is as good as patching);
2. the AUROC of synapse count is within 0.05 of the ceiling (no room left for methods).

If the hypothesis survives, conclusions about a "mechanism" also require the difference of gaps (trained minus
untrained) > 0 with the CI above 0; otherwise patching beats anatomy also where the mechanism of the task is
absent, and we describe it that way.

**Correctness checks (before measuring).** A trained model without ablation reproduces the stored flyvis output
within numerical tolerance; ablating a class with zero synapse count does not change the output; patching
converges at K = 32 (rankings at K = 32 and K = 64 on one model, Spearman >= 0.95, otherwise K = 64).

## Follow-up: is the ceiling of patching hiding the answer? - 2026-11-03

In the first analysis the difference of gaps (trained minus untrained) = 0.091 passed the two-interval rule, but
in untrained networks the AUROC of patching = 1.000, so the untrained gap is practically 1 - AUROC(synapse
count). The mechanism sentence is split into two claims, each tested in a way that does not depend on the
saturation of patching:
- **Claim 1:** the trained network is less predictable from structure (synapse count predicts ablation effects
  worse in trained than in untrained networks), paired by model.
- **Claim 2:** an activity-based method captures this (patching loses less than synapse count after training,
  i.e. the difference of gaps > 0 also when patching in untrained networks is below the oracle).

Identity per model: difference of gaps = change of patching - change of synapse count. A measure variant is
saturated in untrained networks when the mean of patching there is >= 0.99 or >= its ceiling - 0.01; saturated
variants are reported but do not enter the claim 2 decision. The decisive non-saturating measure is the
**top-61 Spearman**: Spearman of the method's score with E within the positive classes (top 10% in that model and
network). Sign rule as below; an inconclusive result with 5 models is absence of evidence, not refutation; only a
result of the opposite sign refutes.

## Sign rule (all annexes)

A quantity is "positive" when the lower bounds of BOTH CIs (bootstrap and t) are > 0; "negative" when the upper
bounds of both are < 0; in every other case (also a bound equal to 0, and nan) "inconclusive". A model whose
measure is nan is left out and listed; a quantity is "undetermined" when at least half of the models it should
be computed on are left out.

## Replication annex (2026-11-09)

Written before any replication number (known at the time: the tables of the discovery analysis for models
000-004; no value for models 005-009 or for the second stimulus set).

**Constants as in the discovery analysis.** Trained and untrained networks (seed of the untrained network = model
number), the 40 stimuli of B and the truth from E on B (top 10%, 61 classes), K = 32, random vectors with seed
[20261101, b], held-out set B', pairing by model identifier, both CIs.

**Models.** All 10: 000-009. Models 000-004 keep their discovery files; models 005-009 ("new") are the next five
by identifier (same selection rule, no selection by results). No exclusions other than the nan rule. Every
quantity has three versions: **all 10** (000-009, the decision that counts), **discovery** (000-004, a
reproduction check: the same numbers as before), **replication** (005-009, models that did not influence the
choice of methods and thresholds).

**Primary decisions (confirmatory, 4 tests).**

| decision | quantity | rule | undetermined |
|---|---|---|---|
| 1. patching beats synapse count | gap in trained networks, all 10 models | "survives": both lower bounds > 0, (ceiling - AUROC(synapse count)) > 0.05 and Holm; "refuted": both upper bounds < 0; otherwise "inconclusive" | >= 5 of 10 missing |
| 2. cheap methods beat synapse count | gradient minus synapse count and activity x weight minus synapse count | "cheap methods survive": both positive and Holm; "partial": one positive; "refuted": none positive and >= 1 negative; otherwise "inconclusive" | as decision 1 for either |
| 3. training lowers the AUROC of synapse count (claim 1) | AUROC(synapse count, trained) - AUROC(synapse count, untrained) | "survives": negative and Holm; "refuted": positive; otherwise "inconclusive" | no untrained pairs for new models |
| 4. patching beats synapse count on the second stimulus set | the gap against E on the second set, models valid on it | as decision 1 with the ceiling of the second set (one half against the other) | >= half of the models without a valid second set |

**Holm correction.** For the four tests, one-sided p values from the t test (df = N - 1; > 0 for decisions 1, 2
and 4, < 0 for decision 3; decision 2 is a conjunction and its p is the larger of the two), Holm at level 0.025
one-sided: sorted p(1) <= ... <= p(4), reject p(k) when p(k) <= 0.025 / (4 - k + 1) and all earlier ones were
rejected. "Survives" needs both rules at once. Holm is computed separately for the all-10 and replication
versions.

**Combination with the discovery analysis.** Discovery decision = the earlier rules (no Holm) on 000-004, then
the all-10 decision and the replication decision. "Confirmed" when the all-10 decision has the sign of the
discovery decision; "weakened" when the discovery decision is decided and the all-10 one inconclusive; "reversed"
when the sign flips; "disputed" when the replication contradicts the all-10 decision, or the all-10 one departs
from discovery while the replication agrees with discovery. Decision 4 has no discovery decision (the second
stimulus set did not exist then), so the all-10 decision is adopted.

**Second stimulus set and its validity test.** Generator flyvis `MovingBar`, width 4, speed 13, intensities 0 and
1, angles 7 + 30k degrees (k = 0..11), 74 frames each; disjoint from B and B' (angle, speed, width, identifier;
tested). Two halves of 12 bars, random with seed 20261101, stratified by intensity. A model is "valid on the
second set" when all four hold (thresholds fixed with no effect on that set known): (i) at least 50% of classes
have E > 0; (ii) the ratio of the 0.9 quantile to the median of E over positive classes is >= 3 and the maximum
is finite; (iii) AUROC(E on one half against the truth from the other) >= 0.95; (iv) the full network responds
(||r||² > 0) to at least 20 of the 24 bars.

**Minimal detectable effects** for describing inconclusive results: 0.12 AUROC with 5 models and about 0.07 with
10 ("no effect larger than MDE" when both CIs lie within [-MDE, MDE]). Power computed after the first analysis
was used only to choose which quantities to compute, never to justify a threshold or a decision.

The annex also lists an exploratory family (ACDC, SAE, transfer between stimulus sets, thresholds of the truth,
class-size baseline, absolute E). Exploratory results never change a confirmatory decision; those that are part
of this package are in [results.md](results.md).

## Deviations

Every change after the preregistration, with its date and reason. Changes made after data are marked as such.

1. **2026-11-01, shuffled network, before any of its ablation results.** Permuting scales across all 604 classes
   made the dynamics diverge in a smoke test (model 000: resting max|r| 1.4e18 and NaN; model 001: max|r| about
   1e4 vs 3.5 to 4.8 in the trained network). Rejecting unstable draws would select permutations similar to the
   trained network, so it was not used. New definition: scales permuted only within classes of the same sign and
   the same postsynaptic cell type; stability criterion: finite r and max|r| <= 10 x the maximum over trained
   models; fallback: permutation within sign.
2. **2026-11-01, shuffled network, second change, still before any of its ablation results.** The
   within-sign-and-target permutation diverged in 5 of 5 models, the fallback in 4 of 5 (trained scales span 7
   orders of magnitude and depend weakly on synapse counts). New definition: permute effective weights w =
   synapse count x scale within sign and postsynaptic type, so each cell type keeps the exact set and sum of its
   input weights of each sign. Stability per model: max|r| <= 10 x max|r| of the same trained model. If 3 or more
   of 5 models are unstable, the shuffled network is reported as undetermined and the hypothesis rests on the
   trained and untrained networks and the convergence of patching. Outcome: 4 of 5 unstable, undetermined.
3. **2026-11-02, failure criterion, before any trained-network result.** (1) Condition 1 means failure also when
   the CI lies entirely below 0: the hypothesis survives condition 1 only when the lower CI bound of the gap is
   > 0. (2) Top 10% of 604 is ceil(60.4) = 61 classes, ties at the threshold included. (3) Distance to the
   ceiling compared with a tolerance of 1e-9 (<= 0.05 + 1e-9), so that exactly 0.05 counts as failure.
4. **2026-11-02, binary truth when the 61st value of E is <= 0, before any trained-network result.** Then only
   classes with E > 0 are positive; counts of positives are reported per model and for B1 and B2. A model with 0
   positives has undetermined AUROC and is left out; with 3 or more such models the trained result is
   undetermined. (Never triggered: every model and network has 61 positives.)
5. **2026-11-02, interval rule, before any trained-network result.** A simulation showed that with 5 models the
   percentile bootstrap covers 0.84 instead of 0.95 and would falsely rescue the hypothesis in 8.3% instead of
   2.5% of cases, while the t interval covers 0.947. Rule: condition 1 survives only when the lower bounds of BOTH
   intervals are > 0; the same for "difference of gaps above 0". The rule is stricter than the original in every
   case. An exact sign-flip permutation test was rejected because with 5 models its smallest one-sided p is
   1/32 = 0.031 > 0.025.
6. **2026-11-09, before the first replication number (validity controls on models 000-004 and discovery files
   only).** The four primary decisions, their rules, thresholds and Holm unchanged; the rest is exploratory.
   (1) Control 1: the flyvis initialization gives each edge the weight sign x N(du, dv) x 0.01 / mean N of the
   class, so the sum of |w| of a class in the untrained network is 0.01 x its number of edges; Spearman(synapse
   count, |w| untrained) = 0.662, Spearman(synapse count, |w| trained) = 0.426. Claim 1 (decision 3) therefore
   reads "the trained network is less predictable from structure than a network initialized from synapse
   counts"; the decision rule is unchanged. A new exploratory quantity permutes synapse counts within classes.
   (2) Control 2: the noise ceiling is the reference for decision 1; ceiling fractions are reported next to AUROC
   (discovery models: patching 0.974, synapse count 0.524), descriptively. (3) Control 3: in untrained networks
   class size has AUROC 0.974 vs synapse count 0.865; in trained networks no trivial baseline comes within 0.02 of
   synapse count (best: minus source depth 0.642 vs 0.762). Class size becomes a second baseline. (4) Control 4:
   absolute E (without dividing by ||r||²) does not change the result (Jaccard of top 10% 0.937, gap 0.227 vs
   0.225); reported descriptively. (5) Control 5: an independent ablation of 30 classes of model 000 reproduces
   the stored E bit for bit (max relative difference 0.0 < 1e-4), so the ablation code stays. (6) The stimulus
   array of the second set is frozen by its sha256
   (`sha256:928a876d4f61666099139f5d88dbdab18a77fc9a692045b6cd89cebee9a09688`); the run stops on a mismatch.
7. **2026-11-09, computing device, before any number for models 001-009 on the second stimulus set and 005-009
   in trained networks.** The replication runs on an NVIDIA GPU (CUDA, torch 2.15) instead of a laptop CPU or
   MPS. Agreement test on model 000: the second stimulus set with E on CUDA gives AUROC 0.991 for patching, 0.750
   for synapse count, gap 0.241, identical ceiling fractions and top 10%; the full trained pair reproduces the
   discovery files (gap 0.239, patching 0.991, synapse count 0.753). Single class x stimulus values differ by at
   most 3.4e-7 absolute (the same order as CPU vs CPU on another machine: float32 and reduction order). Decision
   rules unchanged. Main analysis: 000-004 from the original files, 005-009 from the GPU; a sensitivity analysis
   with all ten models from the GPU (decisions 1, 2 and 4; decision 3 needs untrained networks) gave the same
   decisions; it is not part of this package.
8. **2026-11-12, presentation metrics, after all data and all decisions.** For the public leaderboard every
   method is reported on AUROC (the primary metric, unchanged) and on three secondary metrics: the share of the
   achievable signal, (AUROC - 0.5) / (ceiling - 0.5) per model (the ceiling fraction of deviation 6, now for
   every method and all 10 models); AUPRC (preregistered for synapse count, sum of |weights| and patching in the
   discovery analysis, now for every method and all 10 models); and precision@61, how many of the 61 most
   important classes are in the method's top 61 (new). They were chosen after the results were known, they
   reframe the same rankings and they change no decision.
