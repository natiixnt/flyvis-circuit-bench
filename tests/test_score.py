"""`fcb score` and the leaderboard metrics on synthetic rankings."""

import json

import numpy as np
import pytest

from flyvis_circuit_bench import metrics as M
from flyvis_circuit_bench import score
from flyvis_circuit_bench.cli import main
from flyvis_circuit_bench.data import MODELS_POOLED, load


def test_precision_at_k_counts_overlap_and_splits_ties():
    label = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    assert M.precision_at_k(np.arange(10)[::-1], label) == 1.0
    assert M.precision_at_k(np.arange(10), label) == 0.0
    # top 3 by score: classes 0, 3 and a three-way tie (1, 2, 4) for the last slot -> 1 + 2/3 hits of 3
    s = np.array([9, 5, 5, 8, 5, 0, 0, 0, 0, 0])
    assert M.precision_at_k(s, label) == pytest.approx((1 + 2 / 3) / 3)
    # the value does not depend on the order of the classes
    perm = np.random.default_rng(0).permutation(10)
    assert M.precision_at_k(s[perm], label[perm]) == pytest.approx(M.precision_at_k(s, label))
    assert np.isnan(M.precision_at_k(s, np.zeros(10, dtype=bool)))


def test_signal_share_is_zero_at_chance_and_one_at_the_ceiling():
    assert M.signal_share(0.5, 0.98) == 0.0
    assert M.signal_share(0.98, 0.98) == 1.0
    assert M.signal_share(0.762, 1.0) == pytest.approx(0.524)
    assert np.isnan(M.signal_share(0.7, 0.5))


@pytest.fixture(scope="module")
def data(package):
    return load(package)


def test_inputs_contain_no_truth(data):
    x = score.method_inputs(data)
    assert sorted(x) == list(MODELS_POOLED)
    assert set(x["000"]) == {"source", "target", "synapse_count", "sum_abs_weight", "n_edges"}


def test_evaluate_ranks_the_truth_at_the_ceiling_and_noise_at_chance(data):
    truth = {m: data["models"][m]["B_C1"]["E_B"] for m in MODELS_POOLED}
    best = score.evaluate(truth, data)
    assert best["AUROC"]["pooled"]["mean"] == 1.0 and best["precision_at_61"]["pooled"]["mean"] == 1.0
    assert best["AUROC"]["pooled"]["n"] == 10 and best["AUROC"]["original"]["n"] == 5
    rng = np.random.default_rng(1)
    noise = {m: rng.normal(size=len(truth[m])) for m in MODELS_POOLED}
    r = score.evaluate(noise, data)
    assert abs(r["AUROC"]["pooled"]["mean"] - 0.5) < 0.1
    assert abs(r["signal_share"]["pooled"]["mean"]) < 0.25


def test_evaluate_reproduces_the_leaderboard_row_of_a_builtin_method(data, results):
    s = {m: data["models"][m]["B_C1"]["S"] for m in MODELS_POOLED}
    mine = score.evaluate(s, data)
    ours = next(x for x in results["leaderboard"]["rows"] if x["key"] == "C1|S")
    for metric in ("AUROC", "signal_share", "AUPRC", "precision_at_61"):
        assert mine[metric]["pooled"]["mean"] == pytest.approx(ours[metric]["pooled"]["mean"])


@pytest.mark.parametrize("bad,match", [
    ({"000": np.zeros(3)}, "missing"),
    ({**{m: np.zeros(120) for m in MODELS_POOLED}, "010": np.zeros(120)}, "unknown model"),
    ({m: np.zeros(119) for m in MODELS_POOLED}, "shape"),
    ({m: np.full(120, np.nan) for m in MODELS_POOLED}, "nan or inf"),
    ({m: ["a"] * 120 for m in MODELS_POOLED}, "not numbers"),
])
def test_bad_scores_are_rejected(data, bad, match):
    with pytest.raises(ValueError, match=match):
        score.evaluate(bad, data)


def test_cli_score_prints_a_leaderboard_row(package, tmp_path, capsys):
    d = load(package)
    path = tmp_path / "mine.npz"
    np.savez(path, **{m: d["models"][m]["B_C1"]["A"] for m in MODELS_POOLED})
    out_json = tmp_path / "mine.json"
    assert main(["score", "--data", str(package), "--scores", str(path), "--name", "weights again",
                 "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "| weights again | ? | " in out and "share of signal" in out
    assert json.loads(out_json.read_text())["AUROC"]["pooled"]["n"] == 10
    one = tmp_path / "one.npy"
    np.save(one, d["models"]["000"]["B_C1"]["S"])
    assert main(["score", "--data", str(package), "--scores", str(one)]) == 0


def test_cli_score_reports_bad_input(package, tmp_path, capsys):
    path = tmp_path / "short.npz"
    np.savez(path, **{m: np.zeros(5) for m in MODELS_POOLED})
    assert main(["score", "--data", str(package), "--scores", str(path)]) == 2
    assert "shape" in capsys.readouterr().err
    assert main(["score", "--data", str(package), "--scores", str(tmp_path / "x.csv")]) == 2
    assert main(["score", "--data", str(tmp_path / "nowhere"), "--scores", str(path)]) == 2
