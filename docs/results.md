# Results

All numbers below are produced by `fcb reproduce` from the measurement package and are identical to the tables
of the original study. Notation: mean over models, 95% CI from the percentile bootstrap over models (10 000
resamples, seed 20261101) and from Student t (df = N - 1). Sign rule: "positive" only when the lower bounds of
both intervals are > 0, "negative" only when both upper bounds are < 0, otherwise "inconclusive". Ground truth:
top 10% of the 604 connection classes by the ablation effect E (61 classes per model and network).

Networks: trained = the released flyvis models; untrained = the same networks at flyvis initialization.
Discovery = models 000-004 (analysed first), replication = models 005-009 (analysed after the replication rules
were fixed). Methods: synapse count of the class (anatomy), sum of |weights| of the class, attribution patching
(32 random output directions), gradient (one direction), activity x weight (mean |synaptic current| on held-out
stimuli), activity (sum of |weights| x mean |activity| of the source type on held-out stimuli), SAE features
(sparse autoencoder), class size (number of neuron-to-neuron edges). "truth ceiling" = reliability of the truth
itself: AUROC of E on one half of the stimuli against the truth from the other half. "top-61 Spearman" =
Spearman with E inside the 61 positive classes, the one measure that does not saturate in untrained networks.
The short codes of the measurement files are listed in [methodology.md](methodology.md#glossary).

## 1. Discovery: methods against the ablation truth (models 000-004)

Mean over models with the 95% bootstrap CI.

| network | method | AUROC | AUPRC | Spearman |
|---|---|---|---|---|
| trained | synapse count | 0.762 [0.728, 0.793] | 0.236 [0.203, 0.274] | 0.363 [0.320, 0.395] |
| trained | sum of \|weights\| | 0.879 [0.852, 0.906] | 0.422 [0.355, 0.482] | 0.843 [0.823, 0.863] |
| trained | attribution patching | 0.987 [0.983, 0.991] | 0.880 [0.849, 0.910] | 0.996 [0.995, 0.997] |
| trained | truth ceiling | 1.000 [0.999, 1.000] | 0.997 [0.992, 0.999] | 0.999 [0.998, 0.999] |
| untrained | synapse count | 0.865 [0.859, 0.870] | 0.311 [0.300, 0.320] | 0.562 [0.557, 0.568] |
| untrained | sum of \|weights\| | 0.974 [0.973, 0.976] | 0.864 [0.849, 0.879] | 0.826 [0.823, 0.830] |
| untrained | attribution patching | 1.000 [0.999, 1.000] | 0.997 [0.995, 0.999] | 0.999 [0.999, 0.999] |
| untrained | truth ceiling | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

### Preregistered contrasts

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| AUROC: attribution patching minus synapse count, trained | 0.225 | [0.196, 0.256] | [0.175, 0.276] | 000 0.239; 001 0.186; 002 0.228; 003 0.189; 004 0.285 | 0.186 | 0.285 | positive |
| AUROC: attribution patching minus sum of \|weights\|, trained | 0.108 | [0.084, 0.132] | [0.068, 0.148] | 000 0.084; 001 0.081; 002 0.144; 003 0.088; 004 0.142 | 0.081 | 0.144 | positive |
| AUPRC: attribution patching minus synapse count, trained | 0.644 | [0.597, 0.693] | [0.567, 0.721] | 000 0.733; 001 0.591; 002 0.680; 003 0.586; 004 0.630 | 0.586 | 0.733 | positive |
| AUPRC: attribution patching minus sum of \|weights\|, trained | 0.458 | [0.392, 0.538] | [0.347, 0.569] | 000 0.437; 001 0.448; 002 0.601; 003 0.354; 004 0.450 | 0.354 | 0.601 | positive |
| AUROC: attribution patching minus synapse count, untrained | 0.135 | [0.129, 0.140] | [0.126, 0.144] | 000 0.144; 001 0.129; 002 0.129; 003 0.140; 004 0.132 | 0.129 | 0.144 | positive |
| AUROC: attribution patching minus sum of \|weights\|, untrained | 0.025 | [0.024, 0.027] | [0.023, 0.028] | 000 0.029; 001 0.025; 002 0.026; 003 0.026; 004 0.023 | 0.023 | 0.029 | positive |
| AUPRC: attribution patching minus synapse count, untrained | 0.686 | [0.676, 0.697] | [0.670, 0.703] | 000 0.707; 001 0.675; 002 0.675; 003 0.692; 004 0.683 | 0.675 | 0.707 | positive |
| AUPRC: attribution patching minus sum of \|weights\|, untrained | 0.133 | [0.117, 0.148] | [0.108, 0.157] | 000 0.147; 001 0.129; 002 0.158; 003 0.115; 004 0.114 | 0.114 | 0.158 | positive |
| AUROC: (patching minus synapse count) trained minus untrained | 0.091 | [0.061, 0.122] | [0.039, 0.142] | 000 0.094; 001 0.057; 002 0.099; 003 0.049; 004 0.153 | 0.049 | 0.153 | positive |
| AUROC: (patching minus sum of \|weights\|) trained minus untrained | 0.082 | [0.057, 0.107] | [0.041, 0.124] | 000 0.055; 001 0.057; 002 0.118; 003 0.062; 004 0.120 | 0.055 | 0.120 | positive |

### Preregistered failure criterion (trained networks)

- (1) patching minus synapse count = 0.225, bootstrap [0.196, 0.256], t [0.175, 0.276]; a lower bound <= 0: **no**.
- (2) AUROC of synapse count = 0.762, ceiling = 1.000, distance 0.238 <= 0.05: **no**.
- Decision: **the benchmark survives**.

## 2. Follow-up: is the advantage of patching a mechanism of training? (models 000-004)

In untrained networks the AUROC of patching saturates at 1.000, so the difference of gaps between trained and untrained networks on AUROC is bounded by the ceiling. The preregistered follow-up uses the top-61 Spearman, the only measure that does not saturate there. On it the sign flips. Refuted: training narrows the gap between patching and anatomy, it does not widen it.

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| AUROC: synapse count, trained minus untrained | -0.103 | [-0.138, -0.073] | [-0.157, -0.049] | 000 -0.103; 001 -0.065; 002 -0.112; 003 -0.065; 004 -0.171 | -0.171 | -0.065 | negative |
| AUROC: (patching minus synapse count) trained minus untrained | 0.091 | [0.061, 0.122] | [0.039, 0.142] | 000 0.094; 001 0.057; 002 0.099; 003 0.049; 004 0.153 | 0.049 | 0.153 | positive |
| top-61 Spearman: synapse count, trained minus untrained | 0.248 | [0.159, 0.312] | [0.123, 0.373] | 000 0.311; 001 0.329; 002 0.281; 003 0.078; 004 0.239 | 0.078 | 0.329 | positive |
| top-61 Spearman: (patching minus synapse count) trained minus untrained | -0.705 | [-0.752, -0.660] | [-0.779, -0.630] | 000 -0.651; 001 -0.696; 002 -0.792; 003 -0.735; 004 -0.650 | -0.792 | -0.650 | negative |

## 3. Replication with 10 models and a second stimulus set

Four primary decisions fixed before any number of models 005-009 or of the second stimulus set existed. Holm correction within each version, one-sided level 0.025, m = 4. Discovery = rules of the first analysis on 000-004 (no Holm), all 10 = pooled 000-009 (main estimate), replication = 005-009.

| decision | discovery | all 10 | replication | combination | p all 10 | Holm all 10 | p replication | Holm replication |
|---|---|---|---|---|---|---|---|---|
| patching beats synapse count (trained) | survives | survives | survives | confirmed | 1.5e-08 | yes | 2.0e-04 | yes |
| gradient and activity x weight beat synapse count | cheap methods survive | cheap methods survive | cheap methods survive | confirmed | 1.7e-07 | yes | 6.1e-04 | yes |
| training lowers the AUROC of synapse count | survives | survives | survives | confirmed | 2.5e-05 | yes | 0.006 | yes |
| patching beats synapse count (second stimulus set) | n/a (set added later) | survives | survives | no discovery decision (the decision on all 10 is adopted) | 6.1e-09 | yes | 2.2e-04 | yes |

Validity of the second stimulus set (conditions fixed before any of its effects existed): (i) >= 50% of classes with E > 0, (ii) q90/median >= 3 and finite max, (iii) ceiling AUROC >= 0.95, (iv) >= 20 of 24 bars with nonzero response.

| model | (i) classes E > 0 | (ii) q90/median | (iii) ceiling | (iv) bars | valid |
|---|---|---|---|---|---|
| 000 | 571 | 216.871 | 1.000 | 24 | yes |
| 001 | 586 | 482.776 | 1.000 | 24 | yes |
| 002 | 576 | 861.627 | 1.000 | 24 | yes |
| 003 | 582 | 472.388 | 1.000 | 24 | yes |
| 004 | 574 | 681.017 | 1.000 | 24 | yes |
| 005 | 594 | 348.914 | 1.000 | 24 | yes |
| 006 | 593 | 495.058 | 1.000 | 24 | yes |
| 007 | 587 | 562.306 | 1.000 | 24 | yes |
| 008 | 585 | 1400.059 | 1.000 | 24 | yes |
| 009 | 591 | 1400.543 | 1.000 | 24 | yes |

### Version: all 10 models (000-009)

Decisions: patching beats synapse count (trained): survives; gradient and activity x weight beat synapse count: cheap methods survive; training lowers the AUROC of synapse count: survives; patching beats synapse count (second stimulus set): survives. Ceiling minus AUROC of synapse count: 0.237, on the second stimulus set 0.245 (threshold > 0.05). Models valid on the second stimulus set: 10 of 10.

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| patching minus synapse count | 0.224 | [0.201, 0.248] | [0.195, 0.253] | 000 0.239; 001 0.186; 002 0.228; 003 0.189; 004 0.285; 005 0.163; 006 0.204; 007 0.270; 008 0.211; 009 0.266 | 0.163 | 0.285 | positive |
| gradient minus synapse count | 0.198 | [0.171, 0.226] | [0.164, 0.232] | 000 0.192; 001 0.162; 002 0.214; 003 0.157; 004 0.271; 005 0.121; 006 0.172; 007 0.245; 008 0.198; 009 0.252 | 0.121 | 0.271 | positive |
| activity x weight minus synapse count | 0.176 | [0.153, 0.200] | [0.147, 0.205] | 000 0.210; 001 0.169; 002 0.133; 003 0.126; 004 0.216; 005 0.137; 006 0.177; 007 0.246; 008 0.153; 009 0.196 | 0.126 | 0.246 | positive |
| synapse count, trained minus untrained | -0.103 | [-0.129, -0.077] | [-0.135, -0.070] | 000 -0.103; 001 -0.065; 002 -0.112; 003 -0.065; 004 -0.171; 005 -0.033; 006 -0.079; 007 -0.147; 008 -0.092; 009 -0.158 | -0.171 | -0.033 | negative |
| patching minus synapse count, second stimulus set | 0.230 | [0.208, 0.252] | [0.203, 0.257] | 000 0.241; 001 0.199; 002 0.234; 003 0.206; 004 0.272; 005 0.163; 006 0.214; 007 0.286; 008 0.218; 009 0.263 | 0.163 | 0.286 | positive |
| gradient minus patching (cost of one backward pass) | -0.026 | [-0.033, -0.019] | [-0.035, -0.017] | 000 -0.047; 001 -0.025; 002 -0.014; 003 -0.032; 004 -0.014; 005 -0.042; 006 -0.032; 007 -0.026; 008 -0.013; 009 -0.014 | -0.047 | -0.013 | negative |
| share of signal, patching | 0.974 | [0.970, 0.978] | [0.969, 0.979] | 000 0.983; 001 0.983; 002 0.974; 003 0.968; 004 0.964; 005 0.984; 006 0.970; 007 0.972; 008 0.975; 009 0.966 | 0.964 | 0.984 | positive |
| share of signal, synapse count | 0.526 | [0.475, 0.574] | [0.464, 0.587] | 000 0.506; 001 0.611; 002 0.517; 003 0.590; 004 0.394; 005 0.658; 006 0.563; 007 0.431; 008 0.553; 009 0.434 | 0.394 | 0.658 | positive |
| share of signal, patching, second stimulus set | 0.969 | [0.964, 0.975] | [0.962, 0.976] | 000 0.983; 001 0.981; 002 0.966; 003 0.970; 004 0.959; 005 0.983; 006 0.957; 007 0.969; 008 0.966; 009 0.960 | 0.957 | 0.983 | positive |
| share of signal, synapse count, second stimulus set | 0.510 | [0.463, 0.557] | [0.452, 0.567] | 000 0.500; 001 0.583; 002 0.498; 003 0.557; 004 0.414; 005 0.657; 006 0.529; 007 0.396; 008 0.530; 009 0.433 | 0.396 | 0.657 | positive |

### Version: discovery (000-004)

Decisions: patching beats synapse count (trained): survives; gradient and activity x weight beat synapse count: cheap methods survive; training lowers the AUROC of synapse count: survives; patching beats synapse count (second stimulus set): survives. Ceiling minus AUROC of synapse count: 0.238, on the second stimulus set 0.245 (threshold > 0.05). Models valid on the second stimulus set: 5 of 5.

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| patching minus synapse count | 0.225 | [0.196, 0.256] | [0.175, 0.276] | 000 0.239; 001 0.186; 002 0.228; 003 0.189; 004 0.285 | 0.186 | 0.285 | positive |
| gradient minus synapse count | 0.199 | [0.166, 0.237] | [0.141, 0.257] | 000 0.192; 001 0.162; 002 0.214; 003 0.157; 004 0.271 | 0.157 | 0.271 | positive |
| activity x weight minus synapse count | 0.171 | [0.137, 0.204] | [0.119, 0.223] | 000 0.210; 001 0.169; 002 0.133; 003 0.126; 004 0.216 | 0.126 | 0.216 | positive |
| synapse count, trained minus untrained | -0.103 | [-0.138, -0.073] | [-0.157, -0.049] | 000 -0.103; 001 -0.065; 002 -0.112; 003 -0.065; 004 -0.171 | -0.171 | -0.065 | negative |
| patching minus synapse count, second stimulus set | 0.230 | [0.209, 0.253] | [0.194, 0.267] | 000 0.241; 001 0.199; 002 0.234; 003 0.206; 004 0.272 | 0.199 | 0.272 | positive |
| gradient minus patching (cost of one backward pass) | -0.026 | [-0.037, -0.016] | [-0.043, -0.009] | 000 -0.047; 001 -0.025; 002 -0.014; 003 -0.032; 004 -0.014 | -0.047 | -0.014 | negative |
| share of signal, patching | 0.974 | [0.967, 0.981] | [0.963, 0.985] | 000 0.983; 001 0.983; 002 0.974; 003 0.968; 004 0.964 | 0.964 | 0.983 | positive |
| share of signal, synapse count | 0.524 | [0.455, 0.586] | [0.418, 0.629] | 000 0.506; 001 0.611; 002 0.517; 003 0.590; 004 0.394 | 0.394 | 0.611 | positive |
| share of signal, patching, second stimulus set | 0.971 | [0.964, 0.979] | [0.959, 0.984] | 000 0.983; 001 0.981; 002 0.966; 003 0.970; 004 0.959 | 0.959 | 0.983 | positive |
| share of signal, synapse count, second stimulus set | 0.511 | [0.459, 0.561] | [0.429, 0.592] | 000 0.500; 001 0.583; 002 0.498; 003 0.557; 004 0.414 | 0.414 | 0.583 | positive |

### Version: replication (005-009)

Decisions: patching beats synapse count (trained): survives; gradient and activity x weight beat synapse count: cheap methods survive; training lowers the AUROC of synapse count: survives; patching beats synapse count (second stimulus set): survives. Ceiling minus AUROC of synapse count: 0.236, on the second stimulus set 0.246 (threshold > 0.05). Models valid on the second stimulus set: 5 of 5.

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| patching minus synapse count | 0.223 | [0.189, 0.257] | [0.167, 0.279] | 005 0.163; 006 0.204; 007 0.270; 008 0.211; 009 0.266 | 0.163 | 0.270 | positive |
| gradient minus synapse count | 0.198 | [0.156, 0.238] | [0.130, 0.265] | 005 0.121; 006 0.172; 007 0.245; 008 0.198; 009 0.252 | 0.121 | 0.252 | positive |
| activity x weight minus synapse count | 0.182 | [0.151, 0.217] | [0.129, 0.234] | 005 0.137; 006 0.177; 007 0.246; 008 0.153; 009 0.196 | 0.137 | 0.246 | positive |
| synapse count, trained minus untrained | -0.102 | [-0.141, -0.063] | [-0.166, -0.038] | 005 -0.033; 006 -0.079; 007 -0.147; 008 -0.092; 009 -0.158 | -0.158 | -0.033 | negative |
| patching minus synapse count, second stimulus set | 0.229 | [0.194, 0.267] | [0.169, 0.289] | 005 0.163; 006 0.214; 007 0.286; 008 0.218; 009 0.263 | 0.163 | 0.286 | positive |
| gradient minus patching (cost of one backward pass) | -0.025 | [-0.035, -0.016] | [-0.040, -0.010] | 005 -0.042; 006 -0.032; 007 -0.026; 008 -0.013; 009 -0.014 | -0.042 | -0.013 | negative |
| share of signal, patching | 0.973 | [0.969, 0.979] | [0.965, 0.982] | 005 0.984; 006 0.970; 007 0.972; 008 0.975; 009 0.966 | 0.966 | 0.984 | positive |
| share of signal, synapse count | 0.528 | [0.456, 0.599] | [0.408, 0.647] | 005 0.658; 006 0.563; 007 0.431; 008 0.553; 009 0.434 | 0.431 | 0.658 | positive |
| share of signal, patching, second stimulus set | 0.967 | [0.960, 0.975] | [0.955, 0.979] | 005 0.983; 006 0.957; 007 0.969; 008 0.966; 009 0.960 | 0.957 | 0.983 | positive |
| share of signal, synapse count, second stimulus set | 0.509 | [0.430, 0.587] | [0.383, 0.635] | 005 0.657; 006 0.529; 007 0.396; 008 0.530; 009 0.433 | 0.396 | 0.657 | positive |

## 4. Mean AUROC of every method (descriptive, 10 models)

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| trained: synapse count | 0.763 | [0.737, 0.787] | [0.732, 0.793] | 000 0.753; 001 0.805; 002 0.759; 003 0.794; 004 0.697; 005 0.829; 006 0.781; 007 0.715; 008 0.777; 009 0.717 | 0.697 | 0.829 | positive |
| trained: sum of \|weights\| | 0.878 | [0.855, 0.900] | [0.850, 0.906] | 000 0.907; 001 0.910; 002 0.843; 003 0.895; 004 0.839; 005 0.925; 006 0.892; 007 0.847; 008 0.912; 009 0.812 | 0.812 | 0.925 | positive |
| trained: attribution patching | 0.987 | [0.985, 0.989] | [0.984, 0.989] | 000 0.991; 001 0.992; 002 0.987; 003 0.983; 004 0.982; 005 0.992; 006 0.985; 007 0.986; 008 0.987; 009 0.983 | 0.982 | 0.992 | positive |
| trained: gradient | 0.961 | [0.955, 0.967] | [0.953, 0.969] | 000 0.945; 001 0.967; 002 0.973; 003 0.951; 004 0.968; 005 0.950; 006 0.953; 007 0.960; 008 0.974; 009 0.969 | 0.945 | 0.974 | positive |
| trained: activity x weight | 0.939 | [0.922, 0.955] | [0.918, 0.959] | 000 0.963; 001 0.974; 002 0.892; 003 0.920; 004 0.913; 005 0.965; 006 0.958; 007 0.961; 008 0.929; 009 0.913 | 0.892 | 0.974 | positive |
| trained: activity | 0.936 | [0.918, 0.952] | [0.915, 0.956] | 000 0.954; 001 0.970; 002 0.885; 003 0.930; 004 0.909; 005 0.963; 006 0.957; 007 0.955; 008 0.924; 009 0.909 | 0.885 | 0.970 | positive |
| trained: SAE features | 0.873 | [0.831, 0.904] | [0.828, 0.919] | 000 0.913; 001 0.919; 002 0.834; 003 0.915; 004 0.712; 005 0.912; 006 0.907; 007 0.888; 008 0.864; 009 0.868 | 0.712 | 0.919 | positive |
| trained: class size | 0.601 | [0.572, 0.633] | [0.563, 0.639] | 000 0.712; 001 0.636; 002 0.560; 003 0.543; 004 0.546; 005 0.646; 006 0.615; 007 0.586; 008 0.603; 009 0.564 | 0.543 | 0.712 | positive |
| trained: truth ceiling | 1.000 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 0.999; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 0.999 | 1.000 | positive |
| second stimulus set: synapse count | 0.755 | [0.732, 0.779] | [0.726, 0.784] | 000 0.750; 001 0.791; 002 0.749; 003 0.779; 004 0.707; 005 0.828; 006 0.764; 007 0.698; 008 0.765; 009 0.717 | 0.698 | 0.828 | positive |
| second stimulus set: sum of \|weights\| | 0.876 | [0.854, 0.897] | [0.850, 0.902] | 000 0.902; 001 0.905; 002 0.843; 003 0.901; 004 0.842; 005 0.924; 006 0.877; 007 0.850; 008 0.904; 009 0.816 | 0.816 | 0.924 | positive |
| second stimulus set: attribution patching | 0.985 | [0.982, 0.987] | [0.981, 0.988] | 000 0.991; 001 0.990; 002 0.983; 003 0.985; 004 0.979; 005 0.991; 006 0.978; 007 0.984; 008 0.983; 009 0.980 | 0.978 | 0.991 | positive |
| second stimulus set: gradient | 0.954 | [0.946, 0.963] | [0.944, 0.965] | 000 0.933; 001 0.971; 002 0.970; 003 0.938; 004 0.963; 005 0.948; 006 0.940; 007 0.949; 008 0.970; 009 0.964 | 0.933 | 0.971 | positive |
| second stimulus set: class size | 0.598 | [0.570, 0.632] | [0.559, 0.637] | 000 0.728; 001 0.624; 002 0.564; 003 0.559; 004 0.553; 005 0.643; 006 0.573; 007 0.588; 008 0.588; 009 0.559 | 0.553 | 0.728 | positive |
| second stimulus set: truth ceiling | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

## 5. Validity controls (models 000-004)

Run before the replication data existed, to rule out that the result comes from a shared bug or a trivial explanation (truth and methods share models and code).

### Control 1: the untrained initialization carries synapse-count structure by construction

- Spearman(synapse count, sum of |w| per class, untrained) = 0.662 (identical across models); Spearman(synapse count, edges per class) = 0.662.
- Spearman(synapse count, sum of |w|, trained) = 0.426 [0.402, 0.449] | t [0.387, 0.465].
- Untrained, the sum of |w| of a class equals 0.01 x number of edges (max relative difference 0.000), so the untrained network is "initialized from the connectome", not "unstructured".

### Control 2: noise ceiling of the truth

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| trained: AUROC(E on half 1, truth from half 2) | 1.000 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 0.999; 004 1.000 | 0.999 | 1.000 | positive |
| trained: Spearman(E on half 1, E on half 2) | 0.999 | [0.998, 0.999] | [0.998, 1.000] | 000 0.999; 001 0.999; 002 0.999; 003 0.997; 004 0.999 | 0.997 | 0.999 | positive |
| trained: Jaccard of top 10% on the two halves | 0.944 | [0.896, 0.968] | [0.878, 1.010] | 000 0.968; 001 0.968; 002 0.968; 003 0.848; 004 0.968 | 0.848 | 0.968 | positive |
| trained: share of signal, patching | 0.974 | [0.967, 0.981] | [0.963, 0.985] | 000 0.983; 001 0.983; 002 0.974; 003 0.968; 004 0.964 | 0.964 | 0.983 | positive |
| trained: share of signal, synapse count | 0.524 | [0.455, 0.586] | [0.418, 0.629] | 000 0.506; 001 0.611; 002 0.517; 003 0.590; 004 0.394 | 0.394 | 0.611 | positive |
| untrained: AUROC(E on half 1, truth from half 2) | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000 | 1.000 | 1.000 | positive |
| untrained: Spearman(E on half 1, E on half 2) | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000 | 1.000 | 1.000 | positive |
| untrained: Jaccard of top 10% on the two halves | 0.994 | [0.981, 1.000] | [0.976, 1.011] | 000 1.000; 001 1.000; 002 1.000; 003 0.968; 004 1.000 | 0.968 | 1.000 | positive |
| untrained: share of signal, patching | 0.999 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 0.999; 002 0.999; 003 0.999; 004 1.000 | 0.999 | 1.000 | positive |
| untrained: share of signal, synapse count | 0.730 | [0.719, 0.740] | [0.712, 0.747] | 000 0.711; 001 0.741; 002 0.741; 003 0.719; 004 0.736 | 0.711 | 0.741 | positive |

### Control 3: trivial baselines (AUROC)

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| trained: class size (edges) | 0.599 | [0.547, 0.663] | [0.508, 0.691] | 000 0.712; 001 0.636; 002 0.560; 003 0.543; 004 0.546 | 0.543 | 0.712 | positive |
| trained: source out-degree | 0.602 | [0.582, 0.629] | [0.563, 0.640] | 000 0.604; 001 0.654; 002 0.593; 003 0.574; 004 0.584 | 0.574 | 0.654 | positive |
| trained: source in-degree | 0.570 | [0.540, 0.601] | [0.521, 0.619] | 000 0.603; 001 0.619; 002 0.556; 003 0.548; 004 0.524 | 0.524 | 0.619 | positive |
| trained: target out-degree | 0.596 | [0.564, 0.627] | [0.545, 0.647] | 000 0.538; 001 0.578; 002 0.594; 003 0.630; 004 0.640 | 0.538 | 0.640 | positive |
| trained: target in-degree | 0.531 | [0.505, 0.561] | [0.487, 0.576] | 000 0.487; 001 0.525; 002 0.514; 003 0.582; 004 0.547 | 0.487 | 0.582 | positive |
| trained: minus source depth | 0.642 | [0.613, 0.671] | [0.596, 0.687] | 000 0.604; 001 0.673; 002 0.685; 003 0.609; 004 0.639 | 0.604 | 0.685 | positive |
| trained: minus target depth | 0.540 | [0.509, 0.571] | [0.488, 0.592] | 000 0.483; 001 0.522; 002 0.582; 003 0.533; 004 0.579 | 0.483 | 0.582 | positive |
| trained: synapse count | 0.762 | [0.728, 0.793] | [0.709, 0.814] | 000 0.753; 001 0.805; 002 0.759; 003 0.794; 004 0.697 | 0.697 | 0.805 | positive |
| untrained: class size (edges) | 0.974 | [0.973, 0.976] | [0.972, 0.977] | 000 0.971; 001 0.975; 002 0.974; 003 0.974; 004 0.977 | 0.971 | 0.977 | positive |
| untrained: source out-degree | 0.626 | [0.611, 0.640] | [0.602, 0.649] | 000 0.616; 001 0.645; 002 0.627; 003 0.600; 004 0.642 | 0.600 | 0.645 | positive |
| untrained: source in-degree | 0.688 | [0.677, 0.698] | [0.671, 0.705] | 000 0.683; 001 0.704; 002 0.693; 003 0.668; 004 0.692 | 0.668 | 0.704 | positive |
| untrained: target out-degree | 0.510 | [0.508, 0.512] | [0.506, 0.514] | 000 0.507; 001 0.515; 002 0.507; 003 0.511; 004 0.511 | 0.507 | 0.515 | positive |
| untrained: target in-degree | 0.514 | [0.510, 0.520] | [0.506, 0.523] | 000 0.515; 001 0.526; 002 0.512; 003 0.508; 004 0.512 | 0.508 | 0.526 | positive |
| untrained: minus source depth | 0.461 | [0.456, 0.467] | [0.453, 0.470] | 000 0.462; 001 0.454; 002 0.470; 003 0.465; 004 0.454 | 0.454 | 0.470 | positive |
| untrained: minus target depth | 0.330 | [0.317, 0.343] | [0.310, 0.351] | 000 0.334; 001 0.308; 002 0.335; 003 0.352; 004 0.323 | 0.308 | 0.352 | positive |
| untrained: synapse count | 0.865 | [0.859, 0.870] | [0.856, 0.874] | 000 0.856; 001 0.871; 002 0.871; 003 0.859; 004 0.868 | 0.856 | 0.871 | positive |

### Control 4: normalization of E

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| trained: Jaccard top 10% (relative E vs absolute E) | 0.937 | [0.912, 0.969] | [0.890, 0.985] | 000 0.906; 001 0.937; 002 0.906; 003 0.937; 004 1.000 | 0.906 | 1.000 | positive |
| trained: patching minus synapse count (AUROC) on the absolute truth | 0.227 | [0.198, 0.257] | [0.177, 0.277] | 000 0.234; 001 0.185; 002 0.239; 003 0.192; 004 0.285 | 0.185 | 0.285 | positive |
| untrained: Jaccard top 10% (relative E vs absolute E) | 0.994 | [0.981, 1.000] | [0.976, 1.011] | 000 1.000; 001 1.000; 002 0.968; 003 1.000; 004 1.000 | 0.968 | 1.000 | positive |
| untrained: patching minus synapse count (AUROC) on the absolute truth | 0.135 | [0.131, 0.141] | [0.127, 0.143] | 000 0.144; 001 0.129; 002 0.132; 003 0.140; 004 0.132 | 0.129 | 0.144 | positive |

### Control 5: independent ablation

- 30 classes of model 000, trained (10 large, 10 medium, 10 small effects), re-ablated by a separate code path (zeroing synapse counts, public flyvis `simulate`, central cells from hex coordinates, separate formula for E).
- Max relative difference of E between the two paths: 0.000; between the package file and the stored control: 0.000 (threshold 0.0001). Passed: **yes**.

## 6. Leaderboard: every method on four metrics (10 models)

AUROC is the preregistered primary metric. Share of signal = (AUROC - 0.5) / (ceiling - 0.5) per model, AUPRC and precision@61 (how many of the 61 most important classes are in the method's top 61) are secondary metrics added after the preregistration; they change no decision.

### Trained networks: AUROC

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| attribution patching, all 10 | 0.987 | [0.985, 0.989] | [0.984, 0.989] | 000 0.991; 001 0.992; 002 0.987; 003 0.983; 004 0.982; 005 0.992; 006 0.985; 007 0.986; 008 0.987; 009 0.983 | 0.982 | 0.992 | positive |
| attribution patching, discovery | 0.987 | [0.983, 0.991] | [0.981, 0.993] | 000 0.991; 001 0.992; 002 0.987; 003 0.983; 004 0.982 | 0.982 | 0.992 | positive |
| attribution patching, replication | 0.987 | [0.984, 0.989] | [0.982, 0.991] | 005 0.992; 006 0.985; 007 0.986; 008 0.987; 009 0.983 | 0.983 | 0.992 | positive |
| gradient, all 10 | 0.961 | [0.955, 0.967] | [0.953, 0.969] | 000 0.945; 001 0.967; 002 0.973; 003 0.951; 004 0.968; 005 0.950; 006 0.953; 007 0.960; 008 0.974; 009 0.969 | 0.945 | 0.974 | positive |
| gradient, discovery | 0.961 | [0.951, 0.970] | [0.946, 0.976] | 000 0.945; 001 0.967; 002 0.973; 003 0.951; 004 0.968 | 0.945 | 0.973 | positive |
| gradient, replication | 0.961 | [0.953, 0.969] | [0.948, 0.974] | 005 0.950; 006 0.953; 007 0.960; 008 0.974; 009 0.969 | 0.950 | 0.974 | positive |
| activity x weight, all 10 | 0.939 | [0.922, 0.955] | [0.918, 0.959] | 000 0.963; 001 0.974; 002 0.892; 003 0.920; 004 0.913; 005 0.965; 006 0.958; 007 0.961; 008 0.929; 009 0.913 | 0.892 | 0.974 | positive |
| activity x weight, discovery | 0.932 | [0.906, 0.959] | [0.889, 0.976] | 000 0.963; 001 0.974; 002 0.892; 003 0.920; 004 0.913 | 0.892 | 0.974 | positive |
| activity x weight, replication | 0.945 | [0.926, 0.962] | [0.917, 0.974] | 005 0.965; 006 0.958; 007 0.961; 008 0.929; 009 0.913 | 0.913 | 0.965 | positive |
| activity, all 10 | 0.936 | [0.918, 0.952] | [0.915, 0.956] | 000 0.954; 001 0.970; 002 0.885; 003 0.930; 004 0.909; 005 0.963; 006 0.957; 007 0.955; 008 0.924; 009 0.909 | 0.885 | 0.970 | positive |
| activity, discovery | 0.929 | [0.903, 0.956] | [0.887, 0.972] | 000 0.954; 001 0.970; 002 0.885; 003 0.930; 004 0.909 | 0.885 | 0.970 | positive |
| activity, replication | 0.942 | [0.922, 0.959] | [0.913, 0.971] | 005 0.963; 006 0.957; 007 0.955; 008 0.924; 009 0.909 | 0.909 | 0.963 | positive |
| sum of \|weights\|, all 10 | 0.878 | [0.855, 0.900] | [0.850, 0.906] | 000 0.907; 001 0.910; 002 0.843; 003 0.895; 004 0.839; 005 0.925; 006 0.892; 007 0.847; 008 0.912; 009 0.812 | 0.812 | 0.925 | positive |
| sum of \|weights\|, discovery | 0.879 | [0.852, 0.906] | [0.836, 0.923] | 000 0.907; 001 0.910; 002 0.843; 003 0.895; 004 0.839 | 0.839 | 0.910 | positive |
| sum of \|weights\|, replication | 0.878 | [0.840, 0.913] | [0.819, 0.936] | 005 0.925; 006 0.892; 007 0.847; 008 0.912; 009 0.812 | 0.812 | 0.925 | positive |
| SAE features, all 10 | 0.873 | [0.831, 0.904] | [0.828, 0.919] | 000 0.913; 001 0.919; 002 0.834; 003 0.915; 004 0.712; 005 0.912; 006 0.907; 007 0.888; 008 0.864; 009 0.868 | 0.712 | 0.919 | positive |
| SAE features, discovery | 0.859 | [0.778, 0.916] | [0.748, 0.970] | 000 0.913; 001 0.919; 002 0.834; 003 0.915; 004 0.712 | 0.712 | 0.919 | positive |
| SAE features, replication | 0.888 | [0.870, 0.905] | [0.861, 0.915] | 005 0.912; 006 0.907; 007 0.888; 008 0.864; 009 0.868 | 0.864 | 0.912 | positive |
| synapse count, all 10 | 0.763 | [0.737, 0.787] | [0.732, 0.793] | 000 0.753; 001 0.805; 002 0.759; 003 0.794; 004 0.697; 005 0.829; 006 0.781; 007 0.715; 008 0.777; 009 0.717 | 0.697 | 0.829 | positive |
| synapse count, discovery | 0.762 | [0.728, 0.793] | [0.709, 0.814] | 000 0.753; 001 0.805; 002 0.759; 003 0.794; 004 0.697 | 0.697 | 0.805 | positive |
| synapse count, replication | 0.764 | [0.728, 0.799] | [0.704, 0.823] | 005 0.829; 006 0.781; 007 0.715; 008 0.777; 009 0.717 | 0.715 | 0.829 | positive |
| class size, all 10 | 0.601 | [0.572, 0.633] | [0.563, 0.639] | 000 0.712; 001 0.636; 002 0.560; 003 0.543; 004 0.546; 005 0.646; 006 0.615; 007 0.586; 008 0.603; 009 0.564 | 0.543 | 0.712 | positive |
| class size, discovery | 0.599 | [0.547, 0.663] | [0.508, 0.691] | 000 0.712; 001 0.636; 002 0.560; 003 0.543; 004 0.546 | 0.543 | 0.712 | positive |
| class size, replication | 0.603 | [0.580, 0.627] | [0.565, 0.641] | 005 0.646; 006 0.615; 007 0.586; 008 0.603; 009 0.564 | 0.564 | 0.646 | positive |
| truth ceiling, all 10 | 1.000 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 0.999; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 0.999 | 1.000 | positive |
| truth ceiling, discovery | 1.000 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 0.999; 004 1.000 | 0.999 | 1.000 | positive |
| truth ceiling, replication | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

### Trained networks: share of signal

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| attribution patching, all 10 | 0.974 | [0.970, 0.978] | [0.969, 0.979] | 000 0.983; 001 0.983; 002 0.974; 003 0.968; 004 0.964; 005 0.984; 006 0.970; 007 0.972; 008 0.975; 009 0.966 | 0.964 | 0.984 | positive |
| attribution patching, discovery | 0.974 | [0.967, 0.981] | [0.963, 0.985] | 000 0.983; 001 0.983; 002 0.974; 003 0.968; 004 0.964 | 0.964 | 0.983 | positive |
| attribution patching, replication | 0.973 | [0.969, 0.979] | [0.965, 0.982] | 005 0.984; 006 0.970; 007 0.972; 008 0.975; 009 0.966 | 0.966 | 0.984 | positive |
| gradient, all 10 | 0.922 | [0.910, 0.934] | [0.907, 0.937] | 000 0.890; 001 0.934; 002 0.946; 003 0.904; 004 0.936; 005 0.900; 006 0.907; 007 0.920; 008 0.949; 009 0.938 | 0.890 | 0.949 | positive |
| gradient, discovery | 0.922 | [0.902, 0.940] | [0.892, 0.952] | 000 0.890; 001 0.934; 002 0.946; 003 0.904; 004 0.936 | 0.890 | 0.946 | positive |
| gradient, replication | 0.923 | [0.907, 0.939] | [0.897, 0.948] | 005 0.900; 006 0.907; 007 0.920; 008 0.949; 009 0.938 | 0.900 | 0.949 | positive |
| activity x weight, all 10 | 0.878 | [0.844, 0.910] | [0.837, 0.919] | 000 0.926; 001 0.949; 002 0.783; 003 0.842; 004 0.826; 005 0.931; 006 0.916; 007 0.922; 008 0.858; 009 0.826 | 0.783 | 0.949 | positive |
| activity x weight, discovery | 0.865 | [0.812, 0.918] | [0.778, 0.952] | 000 0.926; 001 0.949; 002 0.783; 003 0.842; 004 0.826 | 0.783 | 0.949 | positive |
| activity x weight, replication | 0.891 | [0.852, 0.925] | [0.834, 0.948] | 005 0.931; 006 0.916; 007 0.922; 008 0.858; 009 0.826 | 0.826 | 0.931 | positive |
| activity, all 10 | 0.872 | [0.837, 0.904] | [0.831, 0.912] | 000 0.909; 001 0.939; 002 0.769; 003 0.862; 004 0.818; 005 0.926; 006 0.914; 007 0.910; 008 0.849; 009 0.819 | 0.769 | 0.939 | positive |
| activity, discovery | 0.860 | [0.807, 0.912] | [0.775, 0.945] | 000 0.909; 001 0.939; 002 0.769; 003 0.862; 004 0.818 | 0.769 | 0.939 | positive |
| activity, replication | 0.883 | [0.844, 0.918] | [0.825, 0.942] | 005 0.926; 006 0.914; 007 0.910; 008 0.849; 009 0.819 | 0.819 | 0.926 | positive |
| sum of \|weights\|, all 10 | 0.757 | [0.709, 0.801] | [0.701, 0.813] | 000 0.815; 001 0.821; 002 0.687; 003 0.792; 004 0.678; 005 0.850; 006 0.784; 007 0.695; 008 0.825; 009 0.624 | 0.624 | 0.850 | positive |
| sum of \|weights\|, discovery | 0.759 | [0.704, 0.813] | [0.671, 0.846] | 000 0.815; 001 0.821; 002 0.687; 003 0.792; 004 0.678 | 0.678 | 0.821 | positive |
| sum of \|weights\|, replication | 0.755 | [0.681, 0.827] | [0.638, 0.872] | 005 0.850; 006 0.784; 007 0.695; 008 0.825; 009 0.624 | 0.624 | 0.850 | positive |
| SAE features, all 10 | 0.747 | [0.661, 0.808] | [0.656, 0.838] | 000 0.825; 001 0.839; 002 0.669; 003 0.833; 004 0.423; 005 0.825; 006 0.814; 007 0.776; 008 0.728; 009 0.737 | 0.423 | 0.839 | positive |
| SAE features, discovery | 0.718 | [0.556, 0.834] | [0.495, 0.941] | 000 0.825; 001 0.839; 002 0.669; 003 0.833; 004 0.423 | 0.423 | 0.839 | positive |
| SAE features, replication | 0.776 | [0.741, 0.811] | [0.721, 0.831] | 005 0.825; 006 0.814; 007 0.776; 008 0.728; 009 0.737 | 0.728 | 0.825 | positive |
| synapse count, all 10 | 0.526 | [0.475, 0.574] | [0.464, 0.587] | 000 0.506; 001 0.611; 002 0.517; 003 0.590; 004 0.394; 005 0.658; 006 0.563; 007 0.431; 008 0.553; 009 0.434 | 0.394 | 0.658 | positive |
| synapse count, discovery | 0.524 | [0.455, 0.586] | [0.418, 0.629] | 000 0.506; 001 0.611; 002 0.517; 003 0.590; 004 0.394 | 0.394 | 0.611 | positive |
| synapse count, replication | 0.528 | [0.456, 0.599] | [0.408, 0.647] | 005 0.658; 006 0.563; 007 0.431; 008 0.553; 009 0.434 | 0.431 | 0.658 | positive |
| class size, all 10 | 0.202 | [0.143, 0.266] | [0.126, 0.278] | 000 0.423; 001 0.273; 002 0.119; 003 0.085; 004 0.091; 005 0.293; 006 0.229; 007 0.172; 008 0.205; 009 0.129 | 0.085 | 0.423 | positive |
| class size, discovery | 0.198 | [0.095, 0.326] | [0.016, 0.381] | 000 0.423; 001 0.273; 002 0.119; 003 0.085; 004 0.091 | 0.085 | 0.423 | positive |
| class size, replication | 0.206 | [0.159, 0.255] | [0.129, 0.282] | 005 0.293; 006 0.229; 007 0.172; 008 0.205; 009 0.129 | 0.129 | 0.293 | positive |
| truth ceiling, all 10 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, discovery | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, replication | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

### Trained networks: AUPRC

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| attribution patching, all 10 | 0.889 | [0.870, 0.908] | [0.866, 0.913] | 000 0.924; 001 0.893; 002 0.905; 003 0.851; 004 0.830; 005 0.939; 006 0.862; 007 0.896; 008 0.897; 009 0.898 | 0.830 | 0.939 | positive |
| attribution patching, discovery | 0.880 | [0.849, 0.910] | [0.832, 0.929] | 000 0.924; 001 0.893; 002 0.905; 003 0.851; 004 0.830 | 0.830 | 0.924 | positive |
| attribution patching, replication | 0.898 | [0.876, 0.922] | [0.864, 0.933] | 005 0.939; 006 0.862; 007 0.896; 008 0.897; 009 0.898 | 0.862 | 0.939 | positive |
| gradient, all 10 | 0.804 | [0.779, 0.829] | [0.773, 0.835] | 000 0.730; 001 0.784; 002 0.866; 003 0.765; 004 0.801; 005 0.775; 006 0.794; 007 0.820; 008 0.850; 009 0.850 | 0.730 | 0.866 | positive |
| gradient, discovery | 0.789 | [0.751, 0.833] | [0.726, 0.852] | 000 0.730; 001 0.784; 002 0.866; 003 0.765; 004 0.801 | 0.730 | 0.866 | positive |
| gradient, replication | 0.818 | [0.792, 0.844] | [0.776, 0.860] | 005 0.775; 006 0.794; 007 0.820; 008 0.850; 009 0.850 | 0.775 | 0.850 | positive |
| activity x weight, all 10 | 0.667 | [0.589, 0.738] | [0.575, 0.759] | 000 0.795; 001 0.824; 002 0.431; 003 0.669; 004 0.532; 005 0.784; 006 0.698; 007 0.748; 008 0.627; 009 0.563 | 0.431 | 0.824 | positive |
| activity x weight, discovery | 0.650 | [0.519, 0.781] | [0.441, 0.859] | 000 0.795; 001 0.824; 002 0.431; 003 0.669; 004 0.532 | 0.431 | 0.824 | positive |
| activity x weight, replication | 0.684 | [0.614, 0.752] | [0.573, 0.795] | 005 0.784; 006 0.698; 007 0.748; 008 0.627; 009 0.563 | 0.563 | 0.784 | positive |
| activity, all 10 | 0.651 | [0.574, 0.718] | [0.562, 0.739] | 000 0.727; 001 0.802; 002 0.404; 003 0.676; 004 0.523; 005 0.769; 006 0.700; 007 0.726; 008 0.619; 009 0.560 | 0.404 | 0.802 | positive |
| activity, discovery | 0.627 | [0.492, 0.747] | [0.426, 0.827] | 000 0.727; 001 0.802; 002 0.404; 003 0.676; 004 0.523 | 0.404 | 0.802 | positive |
| activity, replication | 0.675 | [0.607, 0.738] | [0.570, 0.779] | 005 0.769; 006 0.700; 007 0.726; 008 0.619; 009 0.560 | 0.560 | 0.769 | positive |
| sum of \|weights\|, all 10 | 0.414 | [0.359, 0.466] | [0.348, 0.479] | 000 0.487; 001 0.445; 002 0.303; 003 0.497; 004 0.380; 005 0.532; 006 0.394; 007 0.291; 008 0.503; 009 0.306 | 0.291 | 0.532 | positive |
| sum of \|weights\|, discovery | 0.422 | [0.355, 0.482] | [0.322, 0.523] | 000 0.487; 001 0.445; 002 0.303; 003 0.497; 004 0.380 | 0.303 | 0.497 | positive |
| sum of \|weights\|, replication | 0.405 | [0.318, 0.493] | [0.268, 0.542] | 005 0.532; 006 0.394; 007 0.291; 008 0.503; 009 0.306 | 0.291 | 0.532 | positive |
| SAE features, all 10 | 0.488 | [0.437, 0.532] | [0.430, 0.546] | 000 0.550; 001 0.549; 002 0.329; 003 0.590; 004 0.388; 005 0.534; 006 0.507; 007 0.479; 008 0.439; 009 0.513 | 0.329 | 0.590 | positive |
| SAE features, discovery | 0.481 | [0.385, 0.566] | [0.338, 0.624] | 000 0.550; 001 0.549; 002 0.329; 003 0.590; 004 0.388 | 0.329 | 0.590 | positive |
| SAE features, replication | 0.494 | [0.463, 0.520] | [0.449, 0.540] | 005 0.534; 006 0.507; 007 0.479; 008 0.439; 009 0.513 | 0.439 | 0.534 | positive |
| synapse count, all 10 | 0.248 | [0.221, 0.274] | [0.215, 0.280] | 000 0.191; 001 0.302; 002 0.224; 003 0.265; 004 0.200; 005 0.310; 006 0.290; 007 0.196; 008 0.271; 009 0.228 | 0.191 | 0.310 | positive |
| synapse count, discovery | 0.236 | [0.203, 0.274] | [0.179, 0.294] | 000 0.191; 001 0.302; 002 0.224; 003 0.265; 004 0.200 | 0.191 | 0.302 | positive |
| synapse count, replication | 0.259 | [0.221, 0.294] | [0.201, 0.317] | 005 0.310; 006 0.290; 007 0.196; 008 0.271; 009 0.228 | 0.196 | 0.310 | positive |
| class size, all 10 | 0.150 | [0.132, 0.169] | [0.127, 0.173] | 000 0.217; 001 0.183; 002 0.125; 003 0.123; 004 0.121; 005 0.165; 006 0.168; 007 0.132; 008 0.143; 009 0.123 | 0.121 | 0.217 | positive |
| class size, discovery | 0.154 | [0.123, 0.191] | [0.099, 0.208] | 000 0.217; 001 0.183; 002 0.125; 003 0.123; 004 0.121 | 0.121 | 0.217 | positive |
| class size, replication | 0.146 | [0.131, 0.162] | [0.122, 0.170] | 005 0.165; 006 0.168; 007 0.132; 008 0.143; 009 0.123 | 0.123 | 0.168 | positive |
| truth ceiling, all 10 | 0.998 | [0.996, 0.999] | [0.995, 1.000] | 000 0.999; 001 0.999; 002 1.000; 003 0.988; 004 0.999; 005 0.998; 006 0.999; 007 0.999; 008 0.999; 009 0.999 | 0.988 | 1.000 | positive |
| truth ceiling, discovery | 0.997 | [0.992, 0.999] | [0.991, 1.003] | 000 0.999; 001 0.999; 002 1.000; 003 0.988; 004 0.999 | 0.988 | 1.000 | positive |
| truth ceiling, replication | 0.999 | [0.998, 0.999] | [0.998, 1.000] | 005 0.998; 006 0.999; 007 0.999; 008 0.999; 009 0.999 | 0.998 | 0.999 | positive |

### Trained networks: precision@61

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| attribution patching, all 10 | 0.816 | [0.790, 0.843] | [0.783, 0.849] | 000 0.852; 001 0.885; 002 0.836; 003 0.754; 004 0.754; 005 0.869; 006 0.803; 007 0.787; 008 0.836; 009 0.787 | 0.754 | 0.885 | positive |
| attribution patching, discovery | 0.816 | [0.770, 0.862] | [0.742, 0.890] | 000 0.852; 001 0.885; 002 0.836; 003 0.754; 004 0.754 | 0.754 | 0.885 | positive |
| attribution patching, replication | 0.816 | [0.790, 0.846] | [0.772, 0.861] | 005 0.869; 006 0.803; 007 0.787; 008 0.836; 009 0.787 | 0.787 | 0.869 | positive |
| gradient, all 10 | 0.715 | [0.682, 0.746] | [0.676, 0.754] | 000 0.623; 001 0.738; 002 0.787; 003 0.656; 004 0.754; 005 0.656; 006 0.754; 007 0.689; 008 0.754; 009 0.738 | 0.623 | 0.787 | positive |
| gradient, discovery | 0.711 | [0.656, 0.764] | [0.626, 0.797] | 000 0.623; 001 0.738; 002 0.787; 003 0.656; 004 0.754 | 0.623 | 0.787 | positive |
| gradient, replication | 0.718 | [0.682, 0.751] | [0.663, 0.773] | 005 0.656; 006 0.754; 007 0.689; 008 0.754; 009 0.738 | 0.656 | 0.754 | positive |
| activity x weight, all 10 | 0.608 | [0.551, 0.662] | [0.540, 0.677] | 000 0.672; 001 0.754; 002 0.459; 003 0.623; 004 0.492; 005 0.689; 006 0.639; 007 0.672; 008 0.557; 009 0.525 | 0.459 | 0.754 | positive |
| activity x weight, discovery | 0.600 | [0.505, 0.695] | [0.447, 0.753] | 000 0.672; 001 0.754; 002 0.459; 003 0.623; 004 0.492 | 0.459 | 0.754 | positive |
| activity x weight, replication | 0.616 | [0.561, 0.672] | [0.527, 0.706] | 005 0.689; 006 0.639; 007 0.672; 008 0.557; 009 0.525 | 0.525 | 0.689 | positive |
| activity, all 10 | 0.598 | [0.544, 0.648] | [0.534, 0.662] | 000 0.672; 001 0.705; 002 0.443; 003 0.623; 004 0.492; 005 0.689; 006 0.623; 007 0.656; 008 0.557; 009 0.525 | 0.443 | 0.705 | positive |
| activity, discovery | 0.587 | [0.498, 0.675] | [0.445, 0.729] | 000 0.672; 001 0.705; 002 0.443; 003 0.623; 004 0.492 | 0.443 | 0.705 | positive |
| activity, replication | 0.610 | [0.557, 0.662] | [0.525, 0.694] | 005 0.689; 006 0.623; 007 0.656; 008 0.557; 009 0.525 | 0.525 | 0.689 | positive |
| sum of \|weights\|, all 10 | 0.428 | [0.372, 0.482] | [0.361, 0.495] | 000 0.475; 001 0.475; 002 0.295; 003 0.541; 004 0.393; 005 0.508; 006 0.410; 007 0.311; 008 0.541; 009 0.328 | 0.295 | 0.541 | positive |
| sum of \|weights\|, discovery | 0.436 | [0.354, 0.502] | [0.319, 0.554] | 000 0.475; 001 0.475; 002 0.295; 003 0.541; 004 0.393 | 0.295 | 0.541 | positive |
| sum of \|weights\|, replication | 0.420 | [0.338, 0.502] | [0.291, 0.548] | 005 0.508; 006 0.410; 007 0.311; 008 0.541; 009 0.328 | 0.311 | 0.541 | positive |
| SAE features, all 10 | 0.474 | [0.425, 0.516] | [0.418, 0.529] | 000 0.541; 001 0.557; 002 0.328; 003 0.525; 004 0.377; 005 0.508; 006 0.541; 007 0.426; 008 0.443; 009 0.492 | 0.328 | 0.557 | positive |
| SAE features, discovery | 0.466 | [0.380, 0.544] | [0.335, 0.596] | 000 0.541; 001 0.557; 002 0.328; 003 0.525; 004 0.377 | 0.328 | 0.557 | positive |
| SAE features, replication | 0.482 | [0.446, 0.518] | [0.423, 0.541] | 005 0.508; 006 0.541; 007 0.426; 008 0.443; 009 0.492 | 0.426 | 0.541 | positive |
| synapse count, all 10 | 0.284 | [0.254, 0.313] | [0.247, 0.320] | 000 0.213; 001 0.295; 002 0.311; 003 0.361; 004 0.246; 005 0.328; 006 0.311; 007 0.246; 008 0.311; 009 0.213 | 0.213 | 0.361 | positive |
| synapse count, discovery | 0.285 | [0.243, 0.328] | [0.214, 0.357] | 000 0.213; 001 0.295; 002 0.311; 003 0.361; 004 0.246 | 0.213 | 0.361 | positive |
| synapse count, replication | 0.282 | [0.239, 0.318] | [0.220, 0.344] | 005 0.328; 006 0.311; 007 0.246; 008 0.311; 009 0.213 | 0.213 | 0.328 | positive |
| class size, all 10 | 0.187 | [0.159, 0.215] | [0.151, 0.222] | 000 0.262; 001 0.230; 002 0.148; 003 0.115; 004 0.164; 005 0.230; 006 0.230; 007 0.131; 008 0.197; 009 0.164 | 0.115 | 0.262 | positive |
| class size, discovery | 0.184 | [0.138, 0.233] | [0.108, 0.259] | 000 0.262; 001 0.230; 002 0.148; 003 0.115; 004 0.164 | 0.115 | 0.262 | positive |
| class size, replication | 0.190 | [0.157, 0.223] | [0.137, 0.243] | 005 0.230; 006 0.230; 007 0.131; 008 0.197; 009 0.164 | 0.131 | 0.230 | positive |
| truth ceiling, all 10 | 0.974 | [0.961, 0.984] | [0.959, 0.989] | 000 0.984; 001 0.984; 002 0.984; 003 0.918; 004 0.984; 005 0.967; 006 0.984; 007 0.967; 008 0.984; 009 0.984 | 0.918 | 0.984 | positive |
| truth ceiling, discovery | 0.970 | [0.944, 0.984] | [0.934, 1.007] | 000 0.984; 001 0.984; 002 0.984; 003 0.918; 004 0.984 | 0.918 | 0.984 | positive |
| truth ceiling, replication | 0.977 | [0.970, 0.984] | [0.966, 0.988] | 005 0.967; 006 0.984; 007 0.967; 008 0.984; 009 0.984 | 0.967 | 0.984 | positive |

### Untrained networks: AUROC

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| synapse count, all 10 | 0.865 | [0.862, 0.869] | [0.861, 0.870] | 000 0.856; 001 0.871; 002 0.871; 003 0.859; 004 0.868; 005 0.862; 006 0.860; 007 0.863; 008 0.869; 009 0.875 | 0.856 | 0.875 | positive |
| synapse count, discovery | 0.865 | [0.859, 0.870] | [0.856, 0.874] | 000 0.856; 001 0.871; 002 0.871; 003 0.859; 004 0.868 | 0.856 | 0.871 | positive |
| synapse count, replication | 0.866 | [0.861, 0.871] | [0.858, 0.873] | 005 0.862; 006 0.860; 007 0.863; 008 0.869; 009 0.875 | 0.860 | 0.875 | positive |
| sum of \|weights\|, all 10 | 0.975 | [0.974, 0.976] | [0.974, 0.976] | 000 0.971; 001 0.975; 002 0.974; 003 0.974; 004 0.977; 005 0.976; 006 0.977; 007 0.975; 008 0.975; 009 0.976 | 0.971 | 0.977 | positive |
| sum of \|weights\|, discovery | 0.974 | [0.973, 0.976] | [0.972, 0.977] | 000 0.971; 001 0.975; 002 0.974; 003 0.974; 004 0.977 | 0.971 | 0.977 | positive |
| sum of \|weights\|, replication | 0.976 | [0.975, 0.976] | [0.975, 0.977] | 005 0.976; 006 0.977; 007 0.975; 008 0.975; 009 0.976 | 0.975 | 0.977 | positive |
| attribution patching, all 10 | 1.000 | [1.000, 1.000] | [0.999, 1.000] | 000 1.000; 001 1.000; 002 0.999; 003 0.999; 004 1.000; 005 0.999; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 0.999 | 1.000 | positive |
| attribution patching, discovery | 1.000 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 1.000; 002 0.999; 003 0.999; 004 1.000 | 0.999 | 1.000 | positive |
| attribution patching, replication | 1.000 | [1.000, 1.000] | [0.999, 1.000] | 005 0.999; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 0.999 | 1.000 | positive |
| truth ceiling, all 10 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, discovery | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, replication | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

### Untrained networks: share of signal

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| synapse count, all 10 | 0.730 | [0.723, 0.738] | [0.722, 0.739] | 000 0.711; 001 0.741; 002 0.741; 003 0.719; 004 0.736; 005 0.723; 006 0.721; 007 0.725; 008 0.738; 009 0.750 | 0.711 | 0.750 | positive |
| synapse count, discovery | 0.730 | [0.719, 0.740] | [0.712, 0.747] | 000 0.711; 001 0.741; 002 0.741; 003 0.719; 004 0.736 | 0.711 | 0.741 | positive |
| synapse count, replication | 0.731 | [0.723, 0.742] | [0.716, 0.746] | 005 0.723; 006 0.721; 007 0.725; 008 0.738; 009 0.750 | 0.721 | 0.750 | positive |
| sum of \|weights\|, all 10 | 0.950 | [0.948, 0.952] | [0.948, 0.952] | 000 0.943; 001 0.950; 002 0.947; 003 0.947; 004 0.954; 005 0.952; 006 0.953; 007 0.950; 008 0.951; 009 0.952 | 0.943 | 0.954 | positive |
| sum of \|weights\|, discovery | 0.948 | [0.945, 0.951] | [0.943, 0.953] | 000 0.943; 001 0.950; 002 0.947; 003 0.947; 004 0.954 | 0.943 | 0.954 | positive |
| sum of \|weights\|, replication | 0.952 | [0.950, 0.953] | [0.950, 0.953] | 005 0.952; 006 0.953; 007 0.950; 008 0.951; 009 0.952 | 0.950 | 0.953 | positive |
| attribution patching, all 10 | 0.999 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 0.999; 002 0.999; 003 0.999; 004 1.000; 005 0.999; 006 0.999; 007 1.000; 008 1.000; 009 0.999 | 0.999 | 1.000 | positive |
| attribution patching, discovery | 0.999 | [0.999, 1.000] | [0.999, 1.000] | 000 1.000; 001 0.999; 002 0.999; 003 0.999; 004 1.000 | 0.999 | 1.000 | positive |
| attribution patching, replication | 0.999 | [0.999, 1.000] | [0.999, 1.000] | 005 0.999; 006 0.999; 007 1.000; 008 1.000; 009 0.999 | 0.999 | 1.000 | positive |
| truth ceiling, all 10 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, discovery | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, replication | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

### Untrained networks: AUPRC

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| synapse count, all 10 | 0.313 | [0.306, 0.320] | [0.304, 0.322] | 000 0.293; 001 0.323; 002 0.320; 003 0.302; 004 0.315; 005 0.306; 006 0.303; 007 0.309; 008 0.321; 009 0.337 | 0.293 | 0.337 | positive |
| synapse count, discovery | 0.311 | [0.300, 0.320] | [0.295, 0.326] | 000 0.293; 001 0.323; 002 0.320; 003 0.302; 004 0.315 | 0.293 | 0.323 | positive |
| synapse count, replication | 0.315 | [0.306, 0.327] | [0.298, 0.332] | 005 0.306; 006 0.303; 007 0.309; 008 0.321; 009 0.337 | 0.303 | 0.337 | positive |
| sum of \|weights\|, all 10 | 0.875 | [0.864, 0.884] | [0.862, 0.888] | 000 0.852; 001 0.868; 002 0.837; 003 0.880; 004 0.885; 005 0.884; 006 0.890; 007 0.884; 008 0.894; 009 0.877 | 0.837 | 0.894 | positive |
| sum of \|weights\|, discovery | 0.864 | [0.849, 0.879] | [0.840, 0.889] | 000 0.852; 001 0.868; 002 0.837; 003 0.880; 004 0.885 | 0.837 | 0.885 | positive |
| sum of \|weights\|, replication | 0.886 | [0.881, 0.891] | [0.878, 0.894] | 005 0.884; 006 0.890; 007 0.884; 008 0.894; 009 0.877 | 0.877 | 0.894 | positive |
| attribution patching, all 10 | 0.997 | [0.996, 0.998] | [0.996, 0.998] | 000 1.000; 001 0.997; 002 0.995; 003 0.994; 004 0.998; 005 0.994; 006 0.997; 007 0.998; 008 0.999; 009 0.997 | 0.994 | 1.000 | positive |
| attribution patching, discovery | 0.997 | [0.995, 0.999] | [0.994, 1.000] | 000 1.000; 001 0.997; 002 0.995; 003 0.994; 004 0.998 | 0.994 | 1.000 | positive |
| attribution patching, replication | 0.997 | [0.996, 0.999] | [0.995, 0.999] | 005 0.994; 006 0.997; 007 0.998; 008 0.999; 009 0.997 | 0.994 | 0.999 | positive |
| truth ceiling, all 10 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, discovery | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 000 1.000; 001 1.000; 002 1.000; 003 1.000; 004 1.000 | 1.000 | 1.000 | positive |
| truth ceiling, replication | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

### Untrained networks: precision@61

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| synapse count, all 10 | 0.262 | [0.252, 0.272] | [0.250, 0.275] | 000 0.246; 001 0.279; 002 0.279; 003 0.246; 004 0.246; 005 0.262; 006 0.246; 007 0.262; 008 0.262; 009 0.295 | 0.246 | 0.295 | positive |
| synapse count, discovery | 0.259 | [0.246, 0.272] | [0.237, 0.281] | 000 0.246; 001 0.279; 002 0.279; 003 0.246; 004 0.246 | 0.246 | 0.279 | positive |
| synapse count, replication | 0.266 | [0.252, 0.282] | [0.243, 0.288] | 005 0.262; 006 0.246; 007 0.262; 008 0.262; 009 0.295 | 0.246 | 0.295 | positive |
| sum of \|weights\|, all 10 | 0.808 | [0.797, 0.820] | [0.794, 0.823] | 000 0.770; 001 0.787; 002 0.803; 003 0.820; 004 0.820; 005 0.803; 006 0.803; 007 0.836; 008 0.836; 009 0.803 | 0.770 | 0.836 | positive |
| sum of \|weights\|, discovery | 0.800 | [0.784, 0.816] | [0.773, 0.827] | 000 0.770; 001 0.787; 002 0.803; 003 0.820; 004 0.820 | 0.770 | 0.820 | positive |
| sum of \|weights\|, replication | 0.816 | [0.803, 0.830] | [0.794, 0.839] | 005 0.803; 006 0.803; 007 0.836; 008 0.836; 009 0.803 | 0.803 | 0.836 | positive |
| attribution patching, all 10 | 0.969 | [0.962, 0.975] | [0.960, 0.978] | 000 0.984; 001 0.967; 002 0.967; 003 0.951; 004 0.967; 005 0.951; 006 0.967; 007 0.967; 008 0.984; 009 0.984 | 0.951 | 0.984 | positive |
| attribution patching, discovery | 0.967 | [0.957, 0.977] | [0.953, 0.982] | 000 0.984; 001 0.967; 002 0.967; 003 0.951; 004 0.967 | 0.951 | 0.984 | positive |
| attribution patching, replication | 0.970 | [0.961, 0.980] | [0.953, 0.988] | 005 0.951; 006 0.967; 007 0.967; 008 0.984; 009 0.984 | 0.951 | 0.984 | positive |
| truth ceiling, all 10 | 0.998 | [0.995, 1.000] | [0.995, 1.002] | 000 1.000; 001 1.000; 002 1.000; 003 0.984; 004 1.000; 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 0.984 | 1.000 | positive |
| truth ceiling, discovery | 0.997 | [0.990, 1.000] | [0.988, 1.006] | 000 1.000; 001 1.000; 002 1.000; 003 0.984; 004 1.000 | 0.984 | 1.000 | positive |
| truth ceiling, replication | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 005 1.000; 006 1.000; 007 1.000; 008 1.000; 009 1.000 | 1.000 | 1.000 | positive |

### The README example: mean \|weight\| per edge (trained networks)

The example method of the README section "Add your method" (sum of |weights| / number of edges of each class), scored as `fcb score` does. Not a preregistered method.

| quantity | mean | 95% CI bootstrap | 95% CI t | per model | min | max | sign |
|---|---|---|---|---|---|---|---|
| AUROC, all 10 | 0.827 | [0.809, 0.845] | [0.804, 0.849] | 000 0.772; 001 0.821; 002 0.829; 003 0.879; 004 0.837; 005 0.845; 006 0.824; 007 0.808; 008 0.859; 009 0.793 | 0.772 | 0.879 | positive |
| share of signal, all 10 | 0.654 | [0.618, 0.690] | [0.609, 0.698] | 000 0.544; 001 0.641; 002 0.657; 003 0.759; 004 0.674; 005 0.691; 006 0.649; 007 0.616; 008 0.719; 009 0.587 | 0.544 | 0.759 | positive |
| AUPRC, all 10 | 0.371 | [0.334, 0.409] | [0.324, 0.417] | 000 0.323; 001 0.354; 002 0.320; 003 0.467; 004 0.401; 005 0.426; 006 0.359; 007 0.288; 008 0.461; 009 0.306 | 0.288 | 0.467 | positive |
| precision@61, all 10 | 0.385 | [0.348, 0.425] | [0.338, 0.432] | 000 0.279; 001 0.377; 002 0.410; 003 0.492; 004 0.344; 005 0.426; 006 0.426; 007 0.344; 008 0.443; 009 0.311 | 0.279 | 0.492 | positive |
