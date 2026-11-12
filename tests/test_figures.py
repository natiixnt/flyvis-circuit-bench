"""Figures: all files written, deterministic bytes, readable on a phone, colour-blind safe palette."""

import json
import re

import numpy as np
import pytest

from flyvis_circuit_bench import figures as F
from flyvis_circuit_bench.report import to_json

OKABE_ITO = {"#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"}


@pytest.fixture(scope="module")
def results_json(results):
    return json.loads(to_json(results))


def test_all_figures_written_and_deterministic(results_json, tmp_path):
    a = F.make_all(results_json, tmp_path / "a")
    b = F.make_all(results_json, tmp_path / "b")
    assert len(a) == len(F.FIGURES) * len(F.THEMES) * 2
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes(), pa.name
    assert {p.name for p in a} >= {"hero-light.png", "hero-dark.svg", "replication-light.svg", "per-model-dark.png"}


@pytest.mark.parametrize("name", list(F.FIGURES))
@pytest.mark.parametrize("shown_px,min_px", [(400, 12), (358, 11)])  # 400 px wide; phone 390 px minus margins
def test_text_readable_when_scaled_down(results_json, tmp_path, name, shown_px, min_px):
    """Smallest text, in CSS px, when the saved SVG (tight bounding box) is displayed `shown_px` wide."""
    import re

    plt = F._mpl(F.THEMES["light"])
    fig = F.FIGURES[name](results_json, plt, F.THEMES["light"])
    fig.canvas.draw()
    texts = [t for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize") and hasattr(o, "get_text"))
             if t.get_text().strip()]
    assert texts
    svg = tmp_path / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12)  # as in figures.make_all
    plt.close(fig)
    width_pt = float(re.search(r'<svg[^>]*width="([\d.]+)pt"', svg.read_text()).group(1))
    smallest_px = min(t.get_fontsize() for t in texts) * shown_px / width_pt  # both in pt
    assert smallest_px >= min_px, (name, round(smallest_px, 1))


def test_method_colours_are_okabe_ito():
    for theme in F.THEMES.values():
        assert {theme["S"], theme["P"], theme["A"]} <= OKABE_ITO
        access = theme["access"]
        assert len(set(access)) == len(F.ACCESS) and set(access) <= OKABE_ITO
        assert "#009E73" not in access  # no red-green pair next to vermillion


def test_hero_values_are_the_leaderboard_shares(results_json):
    """Every point, interval and label of the hero is the share of signal of a leaderboard row."""
    from flyvis_circuit_bench.report import pct

    rows = [x for x in results_json["leaderboard"]["rows"] if x["condition"] == "C1" and x["method"] != "ceiling"]
    plt = F._mpl(F.THEMES["light"])
    fig = F.hero(results_json, plt, F.THEMES["light"])
    ax = fig.axes[0]
    points = sorted(float(c.get_offsets()[0][0]) for c in ax.collections)
    labels = sorted(t.get_text() for t in ax.texts)
    intervals = sorted(tuple(map(float, ln.get_xdata())) for ln in ax.lines if len(set(ln.get_ydata())) == 1
                       and ln.get_ydata()[0] >= 0)
    plt.close(fig)
    shares = [x["signal_share"]["pooled"] for x in rows]
    assert points == sorted(s["mean"] for s in shares)
    assert labels == sorted(pct(s["mean"]) for s in shares)
    assert intervals == sorted(tuple(s["ci_bootstrap"]) for s in shares)
    assert len(points) == 8


def _covered(ax, renderer) -> list:
    """Text boxes of the axes that touch a data point, an interval or a reference line."""
    boxes = [t.get_window_extent(renderer) for t in ax.texts]
    points = [p for c in ax.collections for p in ax.transData.transform(c.get_offsets())]
    for line in ax.lines:
        xy = np.asarray(line.get_xydata(), dtype=float)
        disp = line.get_transform().transform(xy)  # axvline: x in data, y in axes coordinates
        for a, b in zip(disp[:-1], disp[1:], strict=True):
            points += list(a + np.linspace(0, 1, 60)[:, None] * (b - a))
    return [(b, p) for b in boxes for p in points if b.expanded(1.05, 1.1).contains(*p)]


def test_hero_labels_do_not_cover_data_or_reference_lines(results_json):
    for theme in F.THEMES.values():
        plt = F._mpl(theme)
        fig = F.hero(results_json, plt, theme)
        fig.canvas.draw()
        assert _covered(fig.axes[0], fig.canvas.get_renderer()) == []
        plt.close(fig)


JARGON = re.compile(r"M_akt|\bM[1-5]\b|\bC[1-3]\b|T5\.|\bF2\b|\bW1\b|[Ww]eek \d|\b[SAP] - [SAP]\b")


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_figure_texts_use_names_not_codes(results_json, name):
    plt = F._mpl(F.THEMES["light"])
    fig = F.FIGURES[name](results_json, plt, F.THEMES["light"])
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_text")) if t.get_text().strip()]
    plt.close(fig)
    bad = [t for t in texts if JARGON.search(t) or t.strip() in {"S", "A", "P"}]
    assert bad == [], bad


def test_per_model_annotations_do_not_cover_points(results_json):
    plt = F._mpl(F.THEMES["light"])
    fig = F.per_model(results_json, plt, F.THEMES["light"])
    fig.canvas.draw()
    ax = fig.axes[0]
    assert len(ax.texts) == 2 and sum(len(c.get_offsets()) for c in ax.collections) == 40
    hits = _covered(ax, fig.canvas.get_renderer())
    plt.close(fig)
    assert hits == []
