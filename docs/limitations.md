# Negative results and limitations

Reported with the same weight as the main result. Numbers come from [results.md](results.md) unless marked
"study only" (measured in the study, not recomputable from this package).

## Negative results

1. **Refuted: training narrows the gap between patching and anatomy, it does not widen it.**
   The original sentence "patching captures what training does and anatomy misses" required the gap between patching and synapse count to be larger in trained than in untrained
   networks. On AUROC the difference of gaps was 0.091 [0.061, 0.122 | t 0.039, 0.142], but in untrained networks
   the AUROC of patching is 1.000, so the contrast is bounded by the ceiling. On the top-61 Spearman, the
   preregistered measure that does not saturate, the difference is -0.705 [-0.752, -0.660 | t -0.779, -0.630],
   negative in all 5 discovery models: the gap between patching and synapse count on Spearman inside the top 61
   is smaller in trained than in untrained networks. What remains is a benchmark statement: patching tracks the ablation truth much better than
   anatomy in both trained and untrained networks.
2. **No cheap proxy replaces patching for free.** The gradient (one backward pass instead of 32) loses
   -0.026 AUROC to patching [-0.033, -0.019 | t -0.035, -0.017] with 10 models; non-inferiority within a margin of
   0.03 was not shown with 5 models (lower t bound -0.043).
3. **This SAE does not beat plain weights.** Its features reach AUROC 0.873 [0.831, 0.904] vs
   sum of |weights| 0.878 [0.855, 0.900] and |weights| x source activity 0.936 (10 models). Study only: SAE features (the codes themselves) did not align
   with cell types, and ACDC (automatic circuit discovery on classes) cost more than patching without beating
   synapse counts. It is one configuration (described in
   [methodology.md](methodology.md#methods)); other dictionary sizes, penalties or inputs were not tried.
4. **The shuffled network is undetermined.** All preregistered permutation schemes made the dynamics diverge in
   4 or 5 of 5 models (study only, smoke tests), so there is no ablation truth for a "wiring kept, weights
   scrambled" network.

## Limitations

- **What "untrained" is.** flyvis initialization is not unstructured: the sum of |weights| of each class is
  exactly 0.01 x its number of edges (validity control 1), and class size alone reaches AUROC 0.974 in untrained
  networks: flyvis initializes every class strength from the connectome (weight sum = 0.01 x number of edges), so
  before training each weight is a fixed function of the anatomy. "Trained vs untrained" means "trained vs
  initialized from the connectome".
- **Zero ablation is the only truth.** The truth sets a class to zero, and patching is a linearization of that
  same zeroing, so part of its AUROC of 0.987 may come from matching the intervention rather than from finding
  what the network relies on. With other interventions (mean ablation, counterfactual inputs) effect sizes can
  change; with mean ablation instead of zeroing the order of the methods holds (v1.1, [mean-ablation.md](v1.1/mean-ablation.md), computed in the research repository and not recomputed by `fcb reproduce`); other interventions are untested.
- **Patching and the truth measure the same thing.** Patching is a first-order estimator of the ablation effect,
  so a high score shows that linearization works in this model. It does not show that patching finds a
  "mechanism", and the benchmark says nothing about function beyond the wiring diagram.
- **One readout.** The truth uses the central column of every cell type; effects that are spatially offset or
  only visible in the task decoder are not captured.
- **Stimuli.** 40 stimuli in the main set (24 edges, 16 Sintel clips) and 24 bars in the second set. The second
  set has no natural movies (the Sintel sequences not used by the main and held-out sets were too few). The
  truth is reliable on these stimuli (ceiling 1.000), which is not the same as complete coverage of the visual
  world.
- **One model family.** One architecture and one trained ensemble (`flow/0000`), 10 of its models. Results for
  other connectomes, tasks or training regimes are untested. Models 000-009 are not a random sample of the 50
  released models: the flyvis documentation states that the model numbers of the pretrained ensemble are
  sorted by task error (0-49, best to worst), so these are the ten best by that ranking and may be better
  trained than a typical model. A second model is the top item of the
  [roadmap](../README.md#roadmap). The conclusions are about this model, not about the fly.
- **The effect of a class varies between models (Z3 †).** The median Spearman correlation of the effect between two
  models is 0.33, and only 3 classes are in the top 61 of at least 9 of the 10 models. The rankings of methods hold on
  the ensemble, because the effects are averaged over models; the effect of one class in one model is not a stable
  quantity, and the conclusions are about the ensemble ([z3.md](v1.1/z3.md)).
- **Neuronal and task truth agree only partly (Z4 †).** The ablation effect and the change of the optic-flow loss
  agree at Spearman 0.667; 34 of the 61 most important classes by ablation effect are not in the top 61 by task
  effect. The task truth rests on 16 Sintel clips and one decoder per model ([z4.md](v1.1/z4.md)).
- **Silencing a type can break the dynamics (Z6 †).** In 12 of 630 type x model pairs the effect of silencing all
  classes of a type is more than 1000 times the median effect of its model, so those conclusions rest on Spearman
  correlations and ranks, not on the size of the effects ([z6.md](v1.1/z6.md)).
- **Small N.** With 5 models the bootstrap alone undercovers (0.84 instead of 0.95), hence the rule that both
  intervals must agree, and Holm correction for the four confirmatory replication tests. Minimal detectable
  effects are about 0.12 AUROC with 5 models and 0.07 with 10.
- **Two devices.** Models 000-004 come from Apple silicon, 005-009 and the second stimulus set from CUDA.
  Differences were at the float32 level (max 3.4e-7 absolute per value) and a sensitivity analysis with one
  device gave the same decisions (study only).
- **Measurement code in this package** covers the truth and every method in the leaderboard (`fcb measure`). The
  second stimulus set has no measurement step in v1.0.0; its outputs are in the measurement package.
- **Secondary metrics chosen after the data.** Share of signal, AUPRC for every method and precision@61 were
  added for the leaderboard after all decisions (deviation 8); only AUROC is preregistered as primary.
- **No external time stamp and no external replication yet.** The preregistration is dated only inside a
  private repository (see [preregistration.md](preregistration.md)); it constrains our analytic freedom but does
  not replace an independent replication. Issues and replications are welcome.
