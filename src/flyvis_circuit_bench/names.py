"""Plain names for the short codes used in the measurement files and in `results/results.json`.

The codes stay in machine-readable places (array keys, JSON keys, file names); everything a person reads
(README, docs, figures, CLI output) uses the names. The one table mapping both is in `docs/methodology.md`.
"""

from __future__ import annotations

METHODS = {
    "S": "synapse count",
    "A": "sum of |weights|",
    "P": "attribution patching",
    "M1": "gradient",
    "M3": "activity x weight",
    "M_akt": "activity",
    "M5": "SAE features",
    "size": "class size",
    "ceiling": "truth ceiling",
    "P_F2": "attribution patching",
    "M1_F2": "gradient",
}

# What a method needs besides the connectome; the truth needs 604 ablations per model and stimulus.
COST = {
    "S": "connectome only",
    "A": "trained weights",
    "P": "32 backward passes",
    "M1": "1 backward pass",
    "M3": "forward pass",
    "M_akt": "forward pass",
    "M5": "forward pass + SAE fit",
    "size": "connectome only",
    "ceiling": "604 ablations",
}

# The access a method needs to the network, from least to most (colour groups of the hero figure).
ACCESS = ("connectome only", "trained weights", "forward pass", "backward pass")
ACCESS_OF = {"size": "connectome only", "S": "connectome only", "A": "trained weights", "M5": "forward pass",
             "M_akt": "forward pass", "M3": "forward pass", "M1": "backward pass", "P": "backward pass"}

CONDITIONS = {"C1": "trained", "C3": "untrained", "F2": "second stimulus set"}
VERSIONS = {"pooled": "all 10 models", "original": "discovery (000-004)", "replication": "replication (005-009)"}
CONDITION_CODES = {"trained": "C1", "untrained": "C3"}

# The preregistered primary decisions of the replication, by their JSON key.
DECISIONS = {
    "T5.1": "patching beats synapse count (trained)",
    "T5.2": "gradient and activity x weight beat synapse count",
    "T5.3": "training lowers the AUROC of synapse count",
    "T5.4": "patching beats synapse count (second stimulus set)",
}

# The one wording of the refuted mechanism hypothesis, used in the README, the docs and docs/results.md.
REFUTED_SENTENCE = "Refuted: training narrows the gap between patching and anatomy, it does not widen it."
