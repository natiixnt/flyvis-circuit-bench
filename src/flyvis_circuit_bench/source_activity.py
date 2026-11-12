"""Two methods that use the recorded activity of the trained network but none of its learned weights.

`activity_sae/C1_<model>.npz` stores `a_typu`: the mean |activity| of each of the 65 cell types on the held-out
stimuli B'. The activity is that of the trained flyvis network (condition C1) driven by those stimuli, not a
recording from a fly. The classes are (source type, target type) pairs, so the activity of a class is the
activity of its source type:

    source activity            a(c) = a_typu[source type of c]
    synapse count x activity   S(c) x a(c), S = number of synapses of the class (the connectome)

Neither reads `A` (sum of |weights|) or any learned parameter. `TYPE_INDEX` gives the position of each source type
in `a_typu` (the flyvis type order); the table was recovered once from the package (the position whose value
equals M_akt / A of every class of that source, identical in all 10 models) and `tests/test_source_activity.py`
checks it against the stored M_akt.
"""

from __future__ import annotations

import numpy as np

TYPE_INDEX = {
    "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6, "R8": 7,
    "L1": 8, "L2": 9, "L3": 10, "L4": 11, "L5": 12, "Lawf1": 13, "Lawf2": 14,
    "Am": 15, "C2": 16, "C3": 17, "CT1(Lo1)": 18, "CT1(M10)": 19,
    "Mi1": 20, "Mi2": 21, "Mi3": 22, "Mi4": 23, "Mi9": 24, "Mi10": 25, "Mi12": 27, "Mi13": 28, "Mi14": 29,
    "Mi15": 30, "T1": 31, "T2": 32, "T2a": 33, "T3": 34, "T4a": 35, "T4b": 36, "T4c": 37, "T4d": 38,
    "T5a": 39, "T5b": 40, "T5c": 41, "T5d": 42, "Tm1": 43, "Tm2": 44, "Tm3": 45, "Tm4": 46, "Tm5Y": 47,
    "Tm5a": 48, "Tm5b": 49, "Tm5c": 50, "Tm9": 51, "Tm16": 52, "Tm20": 53, "Tm28": 54,
    "TmY3": 56, "TmY4": 57, "TmY5a": 58, "TmY9": 59, "TmY10": 60, "TmY13": 61, "TmY14": 62, "TmY15": 63,
    "TmY18": 64,
}


def source_activity(source_types, a_typu) -> np.ndarray:
    """Mean |activity| of the source type of every class (no weights)."""
    a = np.asarray(a_typu, dtype=np.float64)
    return np.array([a[TYPE_INDEX[str(t)]] for t in source_types], dtype=np.float64)


def synapses_x_activity(synapse_count, source_types, a_typu) -> np.ndarray:
    """Synapse count of the class x mean |activity| of its source type (connectome plus recordings)."""
    return np.asarray(synapse_count, dtype=np.float64) * source_activity(source_types, a_typu)
