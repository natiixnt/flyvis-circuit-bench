# Related work

## The same model, the same kind of truth

**FlyHack** (A. Dhawan, "FlyHack: Stress-Testing Mechanistic Claims in a Connectome-Constrained Visual Model",
arXiv:2610.13018, 9 October 2026, https://arxiv.org/abs/2610.13018) ablates every one of the 604 type-to-type
connection classes in flyvis models 000-009 and scores integrated gradients and first-order edge attribution
patching (EAP) against that exhaustive ground truth. In its Table 3 (p. 8), AUROC for the ground-truth top 20
"stays above 0.91 in every row", while precision@20 ranges from 0.44 to 0.66; EAP with zero ablation on both
sides (zero to zero) reaches AUROC 0.927. It also shows that effect sizes are only meaningful for the
intervention the attribution was matched to (zero, held-out mean or counterfactual).

**Exhaustive ablation of the connection classes of flyvis as ground truth is not new here: FlyHack used it
before us.** What this benchmark adds:

- **How much anatomy alone recovers,** on one scale for every method: the share of the achievable signal, from
  0% (chance) to 100% (the truth agreeing with itself across two halves of the stimuli).
- **A ladder of cheap proxies** between anatomy and patching: trained weights, activity, activity x weight,
  this SAE and a single gradient, each with the access to the network it needs.
- **Trained against untrained networks,** with a preregistered mechanism hypothesis that the data refuted
  (negative result 1 in [limitations.md](limitations.md#negative-results)).
- **A benchmark with a leaderboard,** a scoring tool (`fcb score`) and a byte-identical reproduction of every
  table.

**Different readouts, complementary results.** FlyHack measures two behaviours read out by the flyvis decoder:
the decoded flow component along the true direction for a 4 Hz grating, and the negative decoded component for
a contrast-inverting grating (section 6.1, p. 6); the effect of an ablation is the normalized drop of that
number. We measure the response of the central column of every cell type to 40 stimuli (moving edges and
natural movies), and the truth is the top 10% (61 classes). In short (FlyHack: one decoded optical-flow number
per behaviour, truth = top 20, averaged over 20 model x behaviour pairs; here: central column of every cell type,
truth = top 61), patching with zero ablation scores AUROC 0.927 there (Table 3, p. 8) and 0.987 here. A readout that sums the whole circuit into one decoded number and a readout that keeps
every cell type answer different questions; neither result contradicts the other.

## The model

**flyvis** (J. K. Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly
visual system", *Nature* 634, 1132-1140, 2024, https://doi.org/10.1038/s41586-024-07939-3) provides the
connectome-constrained models and the trained ensemble used here.

**Beiran and Litwin-Kumar** ("Prediction of neural activity in connectome-constrained recurrent networks",
*Nature Neuroscience* 28, 2561-2574, 2025, https://doi.org/10.1038/s41593-025-02080-4; preprint bioRxiv 2024,
https://doi.org/10.1101/2024.02.22.581667) argue that "a connectome is often insufficient to constrain the
dynamics of networks that perform a specific task" (preprint abstract). Our results are consistent with this
for flyvis: synapse count alone recovers about half of the achievable signal, and one learned scalar per class
closes about half of the rest.

## Benchmarks with a known answer in language models

- **MIB** (A. Mueller et al., "MIB: A Mechanistic Interpretability Benchmark", ICML 2025,
  https://arxiv.org/abs/2504.13151): circuit localization and causal variable localization across tasks and
  models.
- **InterpBench** (R. Gupta et al., "InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic
  Interpretability Techniques", NeurIPS 2024 Datasets and Benchmarks, https://arxiv.org/abs/2407.14494):
  transformers trained so that a known circuit implements the task.

Both evaluate interpretability methods against a known answer in language models. This benchmark does the same
in a connectome-constrained model of a real nervous system, where the answer is computed by ablation and
anatomy is one of the competitors.
