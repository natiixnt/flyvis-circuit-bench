# Social preview and connectome figure

`docs/figures/social-preview.png` (1280 x 640 px, light theme) and `docs/figures/connectome-{light,dark}.{svg,png}`
are drawn from the measurement package by `fcb figures --data <package>` (`src/flyvis_circuit_bench/connectome.py`).

## What the preview shows

Cell types are the nodes, the 604 connection classes are the edges. Everything is drawn in pale grey and thin, and
two groups are picked out in two colours (no colour scale). The widths of the picked edges follow the synapse count,
`1.0 + 5.0 * sqrt(synapses / max synapses)` points; all other edges have a fixed width.

The two groups are chosen by a fixed rule on how stable a class is across the 10 models (the top 61 is taken per
model, as in Z3). Ties are broken by the class index; each group has 3 classes:

- **thick, little effect** (brown): the 3 classes with the most synapses among the classes that are outside the top 61
  in at least 9 of the 10 models;
- **thin, critical** (blue): the 3 classes with the fewest synapses among the classes that are in the top 61 in at
  least 5 of the 10 models (41 classes).

On the study data these are

| group | classes (source to target), synapses |
|---|---|
| thick, little effect | Tm4 to TmY15 (97 435), L1 to `C3` (53 263), L2 to Tm4 (50 875) |
| thin, critical | L2 to R2 (1 442), L4 to R5 (2 791), R8 to L1 (3 677) |

The widths are drawn from the synapse count, `0.8 + 5.2 * sqrt(synapses / max synapses)` points, and the legend
swatches use the mean drawn width of each group (4.9 pt brown, 1.6 pt blue, about 3 times). Stricter rules were tried
first (blue only from the 3 core classes of Z3, brown also below the median effect in every model); on this data they
give brown lines no thicker than the blue ones, because the core classes have 8 000 to 27 000 synapses.

A test keeps this list in step with the code.

The sentence "Synapse count finds 17 of the 61 connections that matter" is computed, not typed: it is the precision@61 of
the synapse count, mean over the 10 models (`v1_1.z1` table of `results/results.json`), times 61, rounded.

Only the first and the last column carry a title (Retina on the left, Other on the right), so that the direction of
the signal reads at 400 px; the full figure keeps all titles.

## Columns

The columns follow the groups of flyvis itself (`layout.h5` of the connectome and the neuropil labels of its network
figure, `flyvis/analysis/visualization/network_fig.py`): retina; the intermediate types ("lamina, medulla intrinsic
cells, CT1"), split by the family in the type name; the output types ("T-shaped, transmedullary cells"). `Am`, `C2` and `C3`
are intermediate types that the flyvis label does not place, so they stay in "Other". The comments in `LAYERS` give
the source of every group.

## The full figure

`connectome-{light,dark}` keep the continuous colour scale: width = synapse count, colour = mean ablation effect over
the 10 models (log scale, clipped at the 2nd and 98th percentile; cividis, reversed on the light theme so that a large
effect has the contrast), the top 61 outlined, and label boxes for R1 to R6, L1, L2, Mi1, Tm3, T4a and T5a.
