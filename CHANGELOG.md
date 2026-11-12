# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
  `cheap` (gradient, activity x weight), `sae` (activity, SAE features, configuration `SAEConfig`) and
  `structure` (class size); each step writes the file of the measurement package with the same keys.
- Documentation: methodology with a glossary of the study codes, English translation of the preregistration with
  sha256 sums of the original and all deviations, full results, negative results and limitations, related
  work, data dictionary.
