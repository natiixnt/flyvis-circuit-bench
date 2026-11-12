# Methodology

This page describes what is measured and how. The rules for decisions are in
[preregistration.md](preregistration.md), the numbers in [results.md](results.md).

## Model

[flyvis](https://github.com/TuragaLab/flyvis) 1.2.0 (Lappalainen et al., Nature 2024): a recurrent network of
the fly visual system whose architecture is fixed by the connectome (65 cell types, retinotopic hexagonal
columns) and whose parameters were trained on optic flow. We use the released ensemble `flow/0000`, models
000-009 (the first ten by identifier, chosen before any result), with the `best` checkpoint. Models 000-004 are
the **discovery** models (analysed first, they shaped the follow-up questions), models 005-009 the
**replication** models (analysed only after the replication rules were written down).

**Unit of analysis.** A connection class is the set of all synapses from one cell type to another
(type-to-type, 604 classes). Ablating a class means setting its synaptic scale to 0 and leaving every other
parameter unchanged.

## Networks

- **Trained:** the released models.
- **Untrained:** the same configuration built by flyvis without loading a checkpoint. The only random
  parameter of the ensemble configuration is the resting potential (seed = model number). Synaptic scales are
  0.01 / mean synapse count of the class, so the sum of |weights| of a class equals 0.01 x its number of edges:
  the untrained network carries synapse-count structure by construction (validity control 1).
- **Shuffled (undetermined):** trained scales or weights permuted between classes. Every preregistered
  variant made the dynamics diverge in 4 or 5 of 5 models (non-finite or exploding responses), so the shuffled
  network has no ablation truth and is not analysed.

## Stimuli

- **Main set B (40 stimuli):** 24 moving edges from flyvis `MovingEdge` (12 directions x ON and OFF, speed 19,
  width 80) and 16 clips from the MPI Sintel training set rendered into the fly eye as in flyvis
  `RenderedSintel` (3 vertical splits per sequence, 19 frames, resampled to dt = 0.02 s). The list and a random
  split into halves B1 and B2 (seed 20261101) were frozen before the first ablation (`stimuli_B.json` in the
  measurement package).
- **Held-out set B':** a disjoint set used only by the activity-based methods (|weights| x neuron activity, |weights| x source activity, |weights| x source SAE code, source activity only, synapse count x source activity, SAE
  features), so they never see the stimuli of the truth.
- **Second stimulus set (added for the replication):** 24 moving bars (flyvis `MovingBar`, width 4, speed 13,
  angles 7 + 30k degrees, ON and OFF, 74 frames each), disjoint from B and B' in shape, speed, angle and
  identifier, split into two halves stratified by polarity. It has no natural movies.

## Ground truth

The response r(b) is the activity of the central column of every cell type over time (T x 65), after 1 s of grey
input to reach the resting state. For class e:

E(e) = mean over b of ||r_e(b) - r(b)||² / ||r(b)||²

computed on B, B1 and B2 (and on the second stimulus set and its halves). The binary truth is the top 10% of
classes by E (ceil(60.4) = 61 classes; ties at the threshold are included; if the 61st value is <= 0 only classes
with E > 0 count). AUROC measures how well a method's ranking of the 604 classes separates the top 10% by ablation
effect (61 classes).

**Ceiling.** The truth is itself noisy because it depends on the stimuli. Its reliability is the AUROC of E on
B1 against the binary truth from B2 (and Spearman(E_B1, E_B2)).

## Methods

Every method ranks the 604 classes without access to E.

| method | definition | cost |
|---|---|---|
| synapse count | total synapse count of the class (connectome only) | none |
| sum of \|weights\| | sum of \|sign x synapse count x strength\| over the class; the strength is one learned non-negative scalar per class (flyvis `syn_strength`), so this is synapse count times that scalar (parameters, no simulation) | none |
| class size | number of neuron-to-neuron edges in the class | none |
| attribution patching | mean over stimuli b and K = 32 random directions v_k ~ N(0, I) of (s_e x d(v_k . r)/d s_e)² / \|\|r(b)\|\|², an unbiased estimator of the squared first-order change of the output | 1 forward, 32 backward passes per stimulus |
| gradient | the same with one direction, the normalized response itself | 1 forward, 1 backward per stimulus |
| \|weights\| x neuron activity, per edge | mean over B' and time of \|input current through the class\|: sum over the edges of the class of \|weight\| x mean activity of that edge's own source neuron | 1 forward on B' |
| \|weights\| x source activity | sum of \|weights\| of the class x mean \|activity\| of the source cell type on B' (one value per type, so every edge of a class gets the same activity) | 1 forward on B' |
| \|weights\| x source SAE code | sum of \|weights\| of the class x mean L1 norm of the code of the source type; features of this SAE (one configuration, see below) trained on activations on B', mapped to classes | 1 forward + SAE fit |

Random directions use `numpy.random.default_rng([20261101, b])` per stimulus b, so the first k directions do not
depend on K.

**Weights in flyvis.** The weight of an edge is sign x synapse count x strength. Sign and synapse count are fixed
by the connectome (`requires_grad: false` in the model configuration); the strength is trained and shared by all
edges of a class (grouped by source and target type). So within a class, the trained network can only rescale
the anatomy by one scalar, and "sum of |weights|" is synapse count times that scalar.

**This SAE.** One configuration, chosen before its results and not tuned afterwards: a ReLU autoencoder with 640
features (4 x 160) on 4 statistics per neuron and stimulus (mean, standard deviation, maximum and minimum over
time) on B', standardized on the training part; loss = mean squared error + lambda x mean L1 norm of the code;
Adam (learning rate 1e-3), 50 epochs, minibatch 1024, decoder columns normalized after every step, seed
20261101. Lambda is picked per model from {0.001, 0.003, 0.01, 0.03, 0.1} as the largest with fraction of
variance unexplained <= 0.10 on a validation split of 20% of the columns. The score of a class is its sum of
|weights| x the mean L1 norm of the code over the neurons of its source type. Other dictionary sizes, sparsity
penalties or inputs were not tried, so the result is about this SAE, not about sparse autoencoders in general.
The configuration is `flyvis_circuit_bench.measure.methods.SAEConfig` and is stored in the `meta` of every file
`fcb measure --what sae` writes.

## Method inputs

`score.method_inputs(data)` returns, for each of the 10 trained models (`"000"` to `"009"`), a dictionary of
arrays with one entry per connection class (604, in the order of the package):

| field | type | meaning |
|---|---|---|
| `source` | str | presynaptic cell type |
| `target` | str | postsynaptic cell type |
| `synapse_count` | float | total synapse count of the class (connectome) |
| `sum_abs_weight` | float | sum of \|weights\| of the class in the trained model |
| `n_edges` | float | number of neuron-to-neuron edges of the class |

Nothing derived from the ablations is included. Methods that need activations or gradients load the flyvis
models themselves (`uv sync --extra measure`, see [Re-measuring](#re-measuring)).

## Measures

- **AUROC** against the binary truth (Mann-Whitney statistic, ties count one half). The preregistered primary
  metric.
- **AUPRC** (average precision; equal scores form one threshold).
- **Spearman** correlation with the continuous truth E.
- **Top-61 Spearman:** Spearman with E inside the 61 positive classes. In untrained networks the AUROC of
  patching is 1.000, so contrasts that involve untrained networks on AUROC are bounded by the ceiling; this is
  the one measure that does not saturate there.

Added after the preregistration for the leaderboard (deviation 8 in [preregistration.md](preregistration.md#deviations));
they reframe the same rankings and change no decision:

- **Share of the achievable signal** = (AUROC - 0.5) / (ceiling - 0.5), per model, then averaged: 0 is chance,
  1 is as good as the truth agrees with itself. It answers "how much of what can be predicted does a method get".
- **Precision@61:** how many of the 61 most important classes are among the method's top 61 (scores tied at the
  cut share the remaining places in proportion).

Main contrast: the gap AUROC(attribution patching) - AUROC(synapse count) in trained networks. Others: any method
minus synapse count, gradient minus patching (the cost of one backward pass instead of 32), synapse count in
trained minus untrained networks, and the difference of gaps between trained and untrained networks (paired by
model).

## Statistics

- Mean over models, models in sorted order.
- Two 95% intervals: percentile bootstrap over models (10 000 resamples, seed 20261101) and Student t
  (df = N - 1). With 5 models the bootstrap alone covers about 0.84 instead of 0.95 (simulation done before the
  data), which is why a result needs both.
- Sign rule: "positive" when both lower bounds are > 0, "negative" when both upper bounds are < 0, otherwise
  "inconclusive"; "undetermined" when at least half of the planned models are missing or nan.
- Replication: four primary decisions with one-sided t-test p values and Holm correction at 0.025 (m = 4), see
  [preregistration.md](preregistration.md#replication-annex-2026-11-09).

## Validity controls

Before the replication data, five controls on models 000-004 checked that the result is not an artefact of a
shared bug or a trivial explanation: (1) structure of the untrained initialization, (2) noise ceiling of the
truth, (3) trivial graph baselines (class size, degrees, depth from the photoreceptors), (4) normalization of E
(relative vs absolute), (5) an independent ablation of 30 classes by a separate code path (zeroing synapse
counts, public flyvis `simulate`, central cells from hex coordinates, separate formula). Results in
[results.md](results.md#5-validity-controls-models-000-004).

## Devices

Models 000-004 were measured on Apple silicon (CPU and MPS), models 005-009 and every file of the second stimulus
set on an NVIDIA GPU (CUDA, torch 2.15). Before use, model 000 was re-measured on CUDA: AUROC of patching 0.991,
of synapse count 0.753 and gap 0.239 as on the Mac, top 10% identical; single class x stimulus values differed by
at most 3.4e-7 absolute (float32 and reduction order). The pooled analysis uses the Mac files for 000-004, as
preregistered.

## Re-measuring

The measurement path is optional and heavy:

```bash
uv sync --extra measure                    # flyvis 1.2.0 + torch
export FLYVIS_ROOT_DIR=$PWD/flyvis_data    # flyvis downloads the ensemble here; put MPI Sintel in ./sintel
uv run fcb measure --what all --condition trained --model 000 --stimuli data/measurements-v1.0.0/stimuli_B.json
```

`fcb measure` writes the files of the measurement package with the same keys, one step at a time (`--what`):

| step | file | methods | cost |
|---|---|---|---|
| `ablation` | `ablation/C1_000.npz` | truth (E on B, B1, B2, relative effects), synapse count, sum of \|weights\|, attribution patching | 604 ablations x 40 stimuli: 1 to 3 hours on a laptop CPU, about 15 minutes on a recent GPU |
| `cheap` | `cheap_methods/C1_000.npz` | gradient (on B), \|weights\| x neuron activity (on B') | minutes |
| `sae` | `activity_sae/C1_000.npz` | \|weights\| x source activity, \|weights\| x source SAE code (on B'; the SAE configuration is stored in `meta`) | minutes |
| `structure` | `structure.npz` | class size (connectome only) | seconds |

The 40 stimuli of B' ship with the code (`src/flyvis_circuit_bench/measure/stimuli_Bprime.json`, byte-identical
to the study file, sha256 checked, disjoint from B). `cheap` and `sae` exist for trained networks only, as in the
study; the second stimulus set has no measurement step in v1.0.0 (its outputs are in the package). The files
leave out secondary arrays of the study that no table uses (variants of the methods, per-type correlations of the
SAE).

Check against the package, model 000 on Apple silicon (MPS), steps `cheap`, `sae` and `structure` in 6 minutes:
class size is byte-identical. Gradient, |weights| x neuron activity and |weights| x source activity agree within float32 (largest difference
2.8e-6 of the largest score), with the same AUROC to every digit and the same top 61 classes. |weights| x source SAE code agrees
within 1.1e-3 of the largest score (the 50 epochs of the SAE fit amplify float32 differences of its input), AUROC
0.9126 against 0.9125, same top 61. On identical input the code gives the same bytes as the study code.

## Glossary

The study used short codes. They survive in file names, array keys and `results/results.json`; this table is the
only place in the documentation that uses them.

| name in the docs | code | where the code appears |
|---|---|---|
| trained network | `C1` | file names (`C1_000.npz`), JSON keys |
| untrained network (flyvis initialization) | `C3` | file names, JSON keys |
| shuffled network (undetermined) | `C2` | preregistration only |
| synapse count | `S` | array key, JSON keys |
| sum of \|weights\| | `A` | array key, JSON keys |
| attribution patching | `P` (`P_F2` on the second stimulus set) | array keys, JSON keys |
| gradient (one backward pass) | `M1` (`M1_F2`) | `cheap_methods/` |
| \|weights\| x neuron activity | `M3` (`M3v`: its variance) | `cheap_methods/` |
| \|weights\| x source activity | `M_akt` | `activity_sae/` |
| \|weights\| x source SAE code | `M5` | `activity_sae/` |
| mean-substitution patching (models 000-004, not in the tables) | `M2` | `cheap_methods/` |
| ACDC circuit (models 000-004, not in the tables) | `M4` | `activity_sae/` |
| class size | `size`, `n_krawedzi` | JSON keys, `structure.npz` |
| second stimulus set (moving bars) and its halves | `F2`, `F2a`, `F2b` | `f2/`, JSON keys |
| top-61 Spearman | `c1` | JSON keys |
| gap: AUROC(patching) - AUROC(synapse count) | `D_C1` (trained), `D_C3` (untrained), `D^P` | JSON keys |
| a method minus synapse count | `D^M1`, `D^M3`, `D^F2` | JSON keys |
| gradient minus patching | `G^M1` | JSON keys |
| synapse count, trained minus untrained | `Delta S`, `dS` | JSON keys |
| share of the achievable signal | `signal_share`, `fraction` | JSON keys |
| replication decisions 1 to 4 | `T5.1` to `T5.4` | JSON keys |
| validity controls 1 to 5 | `K1` to `K5` | JSON keys, `k5_independent_ablation.json` |
| discovery analysis (models 000-004) and its follow-ups | weeks 1 to 4 | the original Polish preregistration |
| replication analysis (annex of 2026-11-09) | week 5 | the original Polish preregistration |
| the benchmark hypothesis | `W1` | JSON key `W1_failed`, the original Polish preregistration |
| discovery, all 10, replication (versions of a quantity) | `original`, `pooled`, `replication` | JSON keys |
