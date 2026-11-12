# Measurement package: data dictionary

`measurements-v1.0.0.tar.gz` (38.5 MB, CC BY 4.0) is published as a GitHub Release asset. `fcb download`
fetches it, checks the archive against the release `SHA256SUMS`, extracts it to `data/measurements-v1.0.0/`
and checks every file against the `SHA256SUMS` inside. Files are byte-identical to the ones the study tables were
computed from; array keys keep the names used during the study, several of them Polish (translated below).
Short codes of methods and networks in file names and keys are mapped to plain names in the
[glossary](methodology.md#glossary).

Every per-class array has 604 entries in the same class order, given by `zrodlo` (source cell type) and `cel`
(target cell type); `fcb reproduce` refuses files whose class order differs.

## ablation/<condition>_<model>.npz

Trained (`C1_`) and untrained (`C3_`) networks, models 000-009.

| key | shape | meaning |
|---|---|---|
| `E_B`, `E_B1`, `E_B2` | 604 | ablation effect E on stimulus set B and on its halves B1, B2 |
| `wzgledne` | 604 x 40 | relative effect ‖r_e - r‖² / ‖r‖² per class and stimulus ("relative") |
| synapse count | 604 | total synapse count of the class (key in the glossary) |
| sum of \|weights\| | 604 | sum of \|weights\| of the class (key in the glossary) |
| attribution patching, `P_B1`, `P_B2` | 604 | attribution patching (K = 32) on B, B1, B2 |
| `skale` | 604 | synaptic scale of the class ("scales") |
| `r_ref_norma` | 40 | ‖r(b)‖² of the unablated response ("reference norm") |
| `zrodlo`, `cel` | 604 | source and target cell type |
| `bodzce` | 40 | stimulus identifiers ("stimuli"): `krawedz_k<angle>_j<intensity>_v<speed>` = moving edge, `sintel_<sequence>_p<split>` = Sintel clip |
| `meta` | scalar | JSON: condition, model, seed of the condition, device settings (`urzadzenie` = device), timings (`czasy`) |

## cheap_methods/C1_<model>.npz

| key | shape | meaning |
|---|---|---|
| gradient | 604 | attribution patching along one direction (the normalized response) |
| \|weights\| x neuron activity | 604 | mean \|input current through the class\| on held-out stimuli B' |
| `M3v` | 604 | variance of that current (secondary) |
| `M1_b` | 40 x 604 | gradient per stimulus |
| `fbar` | 45669 | mean activity of every neuron on B' |
| `bodzce`, `bodzce_bp` | 40 | identifiers of B and of B' |
| mean-substitution patching, `M2_bk`, `P_alfa`, `Palfa_bk` | | models 000-004 only: mean-substitution patching (discovery analysis, not used in the replication) |

## activity_sae/C1_<model>.npz

| key | shape | meaning |
|---|---|---|
| \|weights\| x source activity | 604 | sum of \|weights\| x mean \|activity\| of the source type on B' |
| \|weights\| x source SAE code | 604 | sum of \|weights\| x mean L1 norm of the sparse autoencoder code of the source type |
| `g`, `a_typu` (mean \|activity\| of each of the 65 cell types on B', the input of the two rows added in v1.0.1), `rho_*`, `n_*`, `AUROC_typu`, `lambda`, `FVU`, `FVU_siatka`, `mu`, `sd`, `n_uczace`, `n_walidacja` | | SAE diagnostics: per-type scores, alignment of features with cell types, chosen sparsity, fraction of variance unexplained on a grid, normalization, train and validation sizes |
| ACDC circuit, `delta`, `obwod`, `kolejnosc`, `H`, `bodzce_M4` | | models 000-004 only: ACDC circuit (discovery analysis; not part of the published tables) |

## f2/C1_<model>.npz

Second stimulus set (24 moving bars), trained networks.

| key | shape | meaning |
|---|---|---|
| `E_F2`, `E_F2a`, `E_F2b` | 604 | ablation effect on the second set and on its halves |
| `wzgledne_F2` | 604 x 24 | relative effect per class and bar |
| `r_ref_norma_F2`, `r_ref_max_abs_F2` | 24 | ‖r‖² and max \|r\| of the unablated response |
| `P_F2`, `P_F2a`, `P_F2b`, `M1_F2`, `M1_F2_b` | | patching and gradient on the second set |
| synapse count, sum of \|weights\| | 604 | as in `ablation/` |
| `bodzce`, `podzial`, `uzyte` | 24 | bar identifiers, half (`F2a` or `F2b`), used flag |

## Other files

| file | content |
|---|---|
| `structure.npz` | `n_krawedzi`: number of neuron-to-neuron edges per class; `zrodlo`, `cel` |
| `k5_independent_ablation.json` | validity control 5: `klasy` (30 class indices), `warstwy` (large, medium, small effects), `E_zapisane` (stored E) and `E_druga_sciezka` (E from the independent path), relative differences |
| `stimuli_B.json` | frozen stimulus set B: generator settings (`krawedzie` = edges, `sintel_render`), list (`bodzce`), halves `B1`, `B2` |
| `MANIFEST.json` | every file with sha256, size and measuring device |
| `README.md` | data card |
| `SHA256SUMS` | sha256 of every file, `sha256sum -c` format |
