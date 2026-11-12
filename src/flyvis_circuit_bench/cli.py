"""Command line interface: `fcb reproduce`, `fcb score`, `fcb download`, `fcb figures`, `fcb verify`, `fcb measure`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

from . import __version__
from .names import CONDITION_CODES

DEFAULT_OWNER = "natiixnt"


def _results(data_dir: Path) -> dict:
    from . import analysis, data

    problems = data.verify(data_dir)
    if problems:
        raise SystemExit("measurement package incomplete or modified:\n  " + "\n  ".join(problems[:20]))
    with warnings.catch_warnings():
        # scipy warns about constant inputs for c1 on single models; those values are nan and reported as such
        warnings.simplefilter("ignore")
        return analysis.analyze(data.load(data_dir))


def cmd_reproduce(a: argparse.Namespace) -> int:
    from . import report

    t0 = time.perf_counter()
    r = _results(a.data)
    outputs = {a.json: report.to_json(r), a.markdown: report.render_markdown(r)}
    if a.check:
        bad = [str(p) for p, text in outputs.items()
               if not p.is_file() or p.read_text(encoding="utf-8") != text]
        if bad:
            print("reproduction differs from the committed files: " + ", ".join(bad), file=sys.stderr)
            return 1
    else:
        for p, text in outputs.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
    from .names import DECISIONS

    h = r["headline"]
    n5, n10 = h["D_C1_N5"], h["D_C1_N10"]
    print(report.main_table(r))
    print(f"AUROC gap, patching minus synapse count, trained, discovery (5 models): {n5['mean']:.3f} "
          f"bootstrap [{n5['ci_bootstrap'][0]:.3f}, {n5['ci_bootstrap'][1]:.3f}] "
          f"t [{n5['ci_t'][0]:.3f}, {n5['ci_t'][1]:.3f}]")
    print(f"the same, all 10 models:                                                {n10['mean']:.3f} "
          f"bootstrap [{n10['ci_bootstrap'][0]:.3f}, {n10['ci_bootstrap'][1]:.3f}] "
          f"t [{n10['ci_t'][0]:.3f}, {n10['ci_t'][1]:.3f}]")
    comb = r["replication"]["combination"]
    print("replication decisions: " + "; ".join(f"{DECISIONS[k]}: {c['d10']} ({c['label']})"
                                                for k, c in comb.items()))
    state = "matches the committed files" if a.check else f"written to {a.json} and {a.markdown}"
    print(f"{state} ({time.perf_counter() - t0:.1f} s)")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    from . import report, score

    try:
        scores = score.load_scores(a.scores)
        result = score.evaluate(scores, _verified(a.data))
    except (OSError, ValueError) as e:
        print(f"cannot score {a.scores}: {e}", file=sys.stderr)
        return 2
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(report.to_json(result), encoding="utf-8")
    print("\n".join(report.LEADERBOARD_HEAD))
    print(score.leaderboard_row(a.name, result, a.cost))
    return 0


def _verified(data_dir: Path) -> dict:
    from . import data

    problems = data.verify(data_dir)
    if problems:
        raise ValueError("measurement package incomplete or modified: " + "; ".join(problems[:5]))
    return data.load(data_dir)


def cmd_download(a: argparse.Namespace) -> int:
    from . import download

    owner = a.owner or os.environ.get("FCB_OWNER") or DEFAULT_OWNER
    if a.archive:
        sums = a.sums.read_text(encoding="utf-8") if a.sums else None
        target = download.install(a.archive, sums, a.dest)
    else:
        target = download.download(owner, a.dest, repo=a.repo, tag=a.tag)
    print(f"verified package in {target}")
    print(f"next: fcb reproduce --data {target}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from . import data

    problems = data.verify(a.data)
    for p in problems:
        print(p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def cmd_figures(a: argparse.Namespace) -> int:
    from . import figures

    results = json.loads(a.results.read_text(encoding="utf-8"))
    for p in figures.make_all(results, a.out):
        print(p)
    return 0


def cmd_measure(a: argparse.Namespace) -> int:
    from .measure import run

    try:
        paths = run.measure(a.what, CONDITION_CODES[a.condition], a.model, a.stimuli, a.out, device=a.device, k=a.k,
                            sintel_dir=a.sintel)
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        print(f"the measurement path needs the [measure] extra (flyvis + torch): {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    for p in paths:
        print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fcb", description=__doc__)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reproduce", help="recompute every result table from the measurement package")
    r.add_argument("--data", type=Path, required=True, help="directory of the extracted measurement package")
    r.add_argument("--json", type=Path, default=Path("results/results.json"))
    r.add_argument("--markdown", type=Path, default=Path("docs/results.md"))
    r.add_argument("--check", action="store_true", help="compare with the existing files instead of writing")
    r.set_defaults(func=cmd_reproduce)

    s = sub.add_parser("score", help="score your own ranking of the 604 classes against the ablation truth")
    s.add_argument("--data", type=Path, required=True, help="directory of the extracted measurement package")
    s.add_argument("--scores", type=Path, required=True,
                   help=".npz with one array of 604 scores per model id (000-009), or one array for all models")
    s.add_argument("--name", default="my method", help="name shown in the leaderboard row")
    s.add_argument("--cost", default="?", help="what the method needs, e.g. '1 backward pass'")
    s.add_argument("--json", type=Path, help="also write every metric and per-model value to this file")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("download", help="download and verify the measurement package from a GitHub Release")
    d.add_argument("--owner", help=f"GitHub account hosting the release (default: FCB_OWNER or {DEFAULT_OWNER})")
    d.add_argument("--repo", default="flyvis-circuit-bench")
    d.add_argument("--tag", default=f"v{__version__}")
    d.add_argument("--dest", type=Path, default=Path("data"))
    d.add_argument("--archive", type=Path, help="install from a local copy of the archive instead")
    d.add_argument("--sums", type=Path, help="release SHA256SUMS for --archive")
    d.set_defaults(func=cmd_download)

    v = sub.add_parser("verify", help="check that a package directory is complete and unmodified")
    v.add_argument("--data", type=Path, required=True)
    v.set_defaults(func=cmd_verify)

    f = sub.add_parser("figures", help="render the figures from results/results.json")
    f.add_argument("--results", type=Path, default=Path("results/results.json"))
    f.add_argument("--out", type=Path, default=Path("docs/figures"))
    f.set_defaults(func=cmd_figures)

    m = sub.add_parser("measure", help="re-measure the truth and every leaderboard method for one model "
                       "(needs the [measure] extra)")
    m.add_argument("--what", choices=("ablation", "cheap", "sae", "structure", "all"), default="ablation",
                   help="ablation: truth, synapse count, sum of |weights|, patching; cheap: gradient, activity x "
                   "weight; sae: activity, SAE features; structure: class size; all: every step")
    m.add_argument("--condition", choices=tuple(CONDITION_CODES), required=True,
                   help="trained (released checkpoint) or untrained (flyvis initialization)")
    m.add_argument("--model", required=True, help="model id of the flyvis ensemble flow/0000, e.g. 000")
    m.add_argument("--stimuli", type=Path, required=True, help="stimuli_B.json from the measurement package")
    m.add_argument("--out", type=Path, default=Path("measurements"))
    m.add_argument("--device", default="cpu")
    m.add_argument("--k", type=int, default=32, help="random output directions for patching")
    m.add_argument("--sintel", type=Path, help="MPI Sintel directory (default: $FLYVIS_ROOT_DIR/sintel)")
    m.set_defaults(func=cmd_measure)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
