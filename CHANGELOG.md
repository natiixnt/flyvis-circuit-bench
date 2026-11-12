# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-11-12

### Fixed

Erratum: v1.0.0 labelled two weight-scaled methods as activity. Names changed, numbers did not:

| v1.0.0 name | v1.0.1 name | AUROC (unchanged) |
|---|---|---|
| activity | \|weights\| x source activity | 0.936 |
| SAE features | \|weights\| x source SAE code | 0.873 |
| activity x weight | \|weights\| x neuron activity, per edge | 0.939 |

- Details: v1.0.0 labelled two weight-scaled methods as activity. The row "activity" is the sum of |weights| of a class
  times the mean |activity| of its source cell type, and "SAE features" is the sum of |weights| times the mean
  L1 norm of the SAE code of the source type, so both use the trained weights. They are now named
  "|weights| x source activity" and "|weights| x source SAE code", with the need "trained weights + forward
  pass" (README, README.pl, figures, docs, leaderboard). "activity x weight" is named "|weights| x neuron
  activity, per edge" (each edge's |weight| times the mean activity of its own source neuron). Method keys in
  `results/results.json` and the measurement package are unchanged, and so are all numbers of v1.0.0.

### Added

- Two leaderboard rows without learned weights, computed from the files of the measurement package (no new
  measurement): "source activity only †" (AUROC 0.715, 43% of the signal) and "synapse count x source activity †"
  (AUROC 0.833, 67%). The activity is that of the trained network on held-out stimuli, not a recording from a
  fly. Marked † (added after the study, not in the preregistered analysis) and listed as deviation 9 of the preregistration.
- Hero figure: a fifth group, "forward pass, no learned weights", with its own colour and marker.

## [1.0.0] - 2026-11-12

### Added

- `fcb reproduce`: every result table (discovery methods and contrasts, the follow-up on the ceiling of
  patching, the replication with Holm correction, method means, validity controls, leaderboard) recomputed from
  the measurement package and checked byte for byte against `results/results.json` and `docs/results.md`.
- `fcb score` and `flyvis_circuit_bench.score`: score your own ranking of the 604 connection classes on AUROC,
  share of the achievable signal, AUPRC and precision@61, in the format of the README leaderboard; notebook
  `notebooks/score_your_method.ipynb` (runs in Colab).
- `fcb download`: measurement package `measurements-v1.0.0.tar.gz` (38.5 MB) from the GitHub Release,
  verified against `SHA256SUMS` at the archive and file level, safe extraction.
- `fcb figures`: hero (share of signal by the access a method needs), per-model and replication figures, PNG
  and SVG, light and dark, colour-blind safe.
- `fcb measure` (optional extra `measure`): measurement of the ablation truth and of every leaderboard method for
  one model with flyvis, in steps `ablation` (truth, synapse count, sum of |weights|, attribution patching),
  `cheap` (gradient, |weights| x neuron activity), `sae` (|weights| x source activity, |weights| x source SAE code, configuration `SAEConfig`) and
  `structure` (class size); each step writes the file of the measurement package with the same keys.
- Documentation: methodology with a glossary of the study codes, English translation of the preregistration with
  sha256 sums of the original and all deviations, full results, negative results and limitations, related
  work, data dictionary.
