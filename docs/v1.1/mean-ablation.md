# Robustness to the intervention: mean ablation instead of zeroing †

† Added after the study, exploratory. This result was computed in the research repository the benchmark comes from
(task 103 there, rules frozen before the measurement), with the same measurement code and the same 10 trained
models. It is **not recomputed by `fcb reproduce`**: the mean-ablation measurements are not in the measurement
package (they need the held-out stimuli B' and a re-run of the model, `fcb measure` does not have this step yet).
The numbers below are copied from the research repository and kept here so that the README can quote them.

## Question

The ablation truth of the study zeroes a connection class, and attribution patching is a linearization of exactly
that. A reviewer can ask whether the high AUROC of patching (0.987) comes from matching the intervention. So the
truth was recomputed with a different intervention.

## Method

- Mean ablation: the current of every edge of the class is replaced by its mean value over held-out stimuli (the
  40 stimuli B', disjoint from the 40 stimuli B of the truth), for the whole run. The readout, the formula of the
  effect and the top-10% binarization (61 classes) are the same as for zeroing.
- Checks before the measurement: zeroing through the same code reproduces the stored effects bit for bit (model
  000, 5 classes); a synthetic network with a class of constant contribution has mean-ablation effect 0 and
  zeroing effect above 0.
- Trained networks, all 10 models. Mean over models with 95% bootstrap interval over models (the t interval gives
  the same picture).
- Rule frozen before the measurement: the ladder of methods holds if the groups (class size and synapse count, then
  |weights|, then forward-pass activity methods, then gradient and patching) rise strictly against the new truth and
  the lower bounds of both intervals of the AUROC of patching stay above 0.9.

## Result

| method | AUROC, zeroing | AUROC, mean ablation | share of signal, zeroing | share of signal, mean ablation |
|---|---|---|---|---|
| attribution patching | 0.987 [0.985, 0.989] | 0.974 [0.971, 0.978] | 97% | 95% |
| gradient | 0.961 [0.955, 0.967] | 0.948 [0.938, 0.957] | 92% | 90% |
| \|weights\| x neuron activity, per edge | 0.939 [0.922, 0.955] | 0.923 [0.907, 0.938] | 88% | 85% |
| \|weights\| x source activity | 0.936 [0.918, 0.952] | 0.925 [0.912, 0.939] | 87% | 85% |
| sum of \|weights\| | 0.878 [0.855, 0.900] | 0.896 [0.881, 0.912] | 76% | 79% |
| \|weights\| x source SAE code | 0.873 [0.831, 0.904] | 0.903 [0.872, 0.928] | 75% | 81% |
| synapse count | 0.763 [0.737, 0.787] | 0.777 [0.760, 0.794] | 53% | 55% |
| class size | 0.601 [0.572, 0.633] | 0.603 [0.574, 0.630] | 20% | 21% |
| truth ceiling | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] | 100% | 100% |
| mean-substitution patching (models 000-004) | 0.963 [0.955, 0.971] | 0.995 [0.994, 0.997] | 93% | 99% |

The two truths agree: Spearman 0.961 [0.954, 0.966] over the 604 classes, Jaccard of the top 10% sets 0.551
[0.527, 0.578]. No class has an infinite response under mean ablation.

**Conclusion †.** The ladder of methods holds against mean ablation: attribution patching loses 0.013 AUROC (0.987
to 0.974), the two methods scaled by activity lose 0.011 to 0.016, while the other methods without a backward
pass (weights, SAE code, synapse count, class size) gain 0.002 to 0.030, so the steps between the rungs get smaller, but
the order of the groups and the bound on patching hold by the rule frozen before the measurement. Mean-substitution
patching, which matches the intervention, reaches 0.995.

Limits: one ensemble of 10 models, one kind of replacement value (the mean over held-out stimuli), the same
readout. The measurements are not part of the package; see the top of this page.
