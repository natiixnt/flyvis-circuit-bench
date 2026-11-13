"""Where the tests look for the real measurement package.

Order: `FCB_DATA`, `data/measurements-v1.1.0` (what `fcb download` writes), `data/wydanie/measurements-v1.1.0` of the
research repository around this directory, then the old package `data/measurements-v1.0.0` (Z4 and Z6 are then not
recomputed). With `REQUIRE_STUDY_DATA=1` only package 1.1.0 is accepted (`test_study_data_is_package_1_1_0`).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _candidates() -> list[Path]:
    out = [ROOT / "data" / "measurements-v1.1.0"]
    if len(ROOT.parents) > 1:
        out.append(ROOT.parents[1] / "data" / "wydanie" / "measurements-v1.1.0")
    return out + [ROOT / "data" / "measurements-v1.0.0"]


def study_data() -> Path:
    if os.environ.get("FCB_DATA"):
        return Path(os.environ["FCB_DATA"])
    found = [p for p in _candidates() if (p / "SHA256SUMS").is_file()]
    return found[0] if found else _candidates()[0]


DATA = study_data()
