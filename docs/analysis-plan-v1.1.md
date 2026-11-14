# Analysis plan for v1.1 (exploratory, added after the study)

Written 2026-11-12, before any of the analyses below was run. Everything here is exploratory and marked † in tables
and figures. Nothing here changes the preregistered replication decisions or how they are computed.

## Shared rules

- Every new metric, row and figure is marked † ("added after the study").
- Whenever something is selected (a threshold, a group, a subset of classes), the selection is made on models
  000-004 only and reported separately on models 005-009 and on all 10 models.
- Statistics as in the study: mean over models, 95% bootstrap interval over models and Student t interval, fixed
  seed 20261101.
- Results go to `results/results.json` under the key `v1_1.<analysis>` and to `docs/v1.1/<analysis>.md`; figures to
  `docs/figures/v1.1-<analysis>-{light,dark}.{svg,png}`. `fcb reproduce --check` reproduces them byte for byte.
- Truth `E(e)` and the binary truth are the study definitions (zeroed synaptic scale of class `e`, central column
  of every cell type, mean over stimuli of `||r_e - r|| / ||r||`), unless a section says otherwise.

## Z1. Sensitivity to the threshold, continuous metrics

- Thresholds: top 5%, 10%, 15%, 20% of 604 classes = top 31, 61, 91, 121 (ceil).
- For every leaderboard method and threshold: AUROC, share of signal `(AUROC - 0.5) / (ceiling - 0.5)` with the
  ceiling recomputed at that threshold, AUPRC, precision@k with k = the threshold count.
- Continuous: Spearman between the method score and `E(e)` over all 604 classes.
- Curve: precision@k for k = 10, 20, ..., 200.
- Ordering change: a pair of methods "swaps" if the sign of the difference of mean AUROC differs between two
  thresholds and both 95% bootstrap intervals of the difference exclude zero at their respective thresholds; pairs
  whose interval includes zero at either threshold are reported as "not separated", not as swaps.

## Z2. Truth split by stimulus family

- Precondition: per-stimulus effects must be stored in the measurement package. If only aggregated effects are
  stored, the analysis stops and is reported as not possible without remeasurement.
- Families: 24 moving-edge stimuli and 16 Sintel clips. `E_edges(e)` and `E_sintel(e)` = mean over the stimuli of
  that family, per model.
- Per model: Spearman(`E_edges`, `E_sintel`), overlap of top 61 (count and Jaccard), split-half ceiling within each
  family (random halves of that family's stimuli, 100 splits, seed 20261101, mean Spearman).
- Full method ladder (AUROC, share of signal) against each family's truth.

## Z3. Agreement between models, variance of learned scalars

- 10x10 matrix of Spearman between models' `E(e)` vectors; top-61 overlap (count, Jaccard) for each pair.
- Per class: mean effect, coefficient of variation of the effect across models (std / mean, classes with mean
  effect 0 excluded and counted), number of models in which the class is in the top 61.
- Groups (fixed here): "core" = in the top 61 in at least 9 of 10 models; "variable" = in the top 61 in 1 to 4 of
  10 models; the rest "other". Group sizes are reported on 000-004, 005-009 and all 10.
- Learned scalar per class per model = sum of |weights| of the class / synapse count of the class; its CV across
  models per class.
- Figure: scalar vs synapse count, colour = ablation effect, top-61 classes highlighted. Spearman between CV of the
  scalar and CV of the effect across classes.
- Docs state that the current ceiling is a split-half over stimuli, not over models, and compare it with the median
  between-model Spearman.

## Z4. Second truth at the task level

- Step 1: check whether decoder outputs of the 604 ablations x 10 models were stored. If yes, compute from stored
  data. If not, remeasure 604 zero ablations x 10 models with the same seeds and store the decoder output.
- Task truth `T(e)`: change of the flyvis training loss of the decoded optic flow on the Sintel clips used in the
  study, ablated minus intact, mean over clips. Binary task truth: top 61 by `T(e)`.
- Results: Spearman(`E`, `T`) and top-61 overlap per model; a 2x2 table of classes high/low on `E` x high/low on
  `T` (high = top 61) with the class lists of the two off-diagonal cells; the full method ladder against `T`.

## Z6 (optional). Silencing cell types vs connections

- Silencing type A = zeroing all classes whose source type is A at once (64 types x 10 models), same truth metric.
- Spearman between the silencing effect of A and the sum and the maximum of the effects of A's outgoing classes.
- "Masking": types whose silencing effect is below 0.5 x the sum of their classes' effects; the reverse: above
  1.5 x the sum. Thresholds fixed here.

## Z7 (optional). Second-order correction for attribution patching

- Patching plus a Hessian-vector-product correction (one extra backward pass per direction). New leaderboard row †.
  Question: does it close the gap from 97% towards 100% of the achievable signal.

Clarification of Z7, 2026-11-13 (before any Z7 measurement; the sentence "one extra backward pass per direction" above
was imprecise, it is one per class and direction):

- Patching `P` (study definition) per stimulus b, readout direction v_k and class c uses the first-order change of
  `v_k . r` when the class scale `s_c` goes to 0: `-s_c g_kc`, with `g_kc = d(v_k . r)/d s_c`, term `(s_c g_kc)^2 / ||r||^2`.
- Corrected change `Delta_k(c) = -s_c g_kc + 0.5 s_c^2 H_{kc,c}`, with `H_{kc,c} = d^2(v_k . r)/d s_c^2`: the diagonal
  entry of the Hessian, taken from the Hessian-vector product with the unit vector `e_c` (double backward, only
  component c of the product is kept). The full Taylor expansion of the ablation `s_c -> 0` along the single coordinate c
  to second order.
- Term `(Delta_k(c))^2 / ||r||^2`, same `r`, same directions `v_k` (same K and seed as `P`), same aggregation as `P`: mean over
  readout directions and stimuli, and over B1 and B2 for the halves. New leaderboard row `P+HVP` †, next to `P`.
- Cost: one HVP per (stimulus, direction, class), so far more than `P`; if the full run is too expensive, the stimulus
  subset or a Hutchinson estimator is proposed with its error before any computation.

Second clarification of Z7, 2026-11-13 (before the Z7 measurement; decided after the cost of one model was measured:
523 s per stimulus on the GPU, about 58 h for 40 stimuli and 10 models):

- The Z7 measurement uses 8 of the 40 stimuli of B, fixed here by a rule independent of any result:
  `numpy.random.default_rng(20261101).choice(40, size=8, replace=False)`, sorted, indices into the order of
  `stimuli_B.json`. This gives indices 4, 12, 16, 18, 20, 27, 37, 39, that is `krawedz_k060_j0_v19`,
  `krawedz_k180_j0_v19`, `krawedz_k240_j0_v19`, `krawedz_k270_j0_v19`, `krawedz_k300_j0_v19`, `sintel_ambush_4_p1`,
  `sintel_temple_2_p0`, `sintel_temple_3_p1`. All 604 classes, all 10 models, K = 32 readout directions with the
  direction of stimulus b taken from the index b in the full set (as in the study).
- Fair comparison: patching `P` is computed on the same 8 stimuli in the same run (the first-order term of the
  corrected change, `first^2 / ||r||^2`), and is the row compared with `P+HVP`; patching of the package (40 stimuli) is
  shown next to it for context only. The truth `E` and its ceiling are those of the study (40 stimuli).
