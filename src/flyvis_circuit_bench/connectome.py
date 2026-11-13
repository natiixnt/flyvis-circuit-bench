"""The connectome as a picture: cell types as nodes in the layers of the visual system, the 604 connection classes
as edges (width = synapse count, colour = mean ablation effect over the 10 models, top 61 outlined).

Computed from the measurement package (`fcb figures --data <package>`), deterministic, vector SVG without embedded
bitmaps. Two products: `connectome-{light,dark}.{svg,png}` (labelled nodes) and `social-preview.png`
(1280 x 640 px, light theme, the title and one sentence on the left, no node labels).

The layer of a cell type is a drawing convention, not data: the neuropil where the type has its main arborisation,
after the conventional anatomy of the fly optic lobe (Fischbach and Dittrich 1989; Takemura et al. 2013). It only
decides the column a type is drawn in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import MODELS_POOLED
from .figures import THEMES, _mpl

# Column of every cell type. Source: the groups of flyvis itself (ensemble connectome `layout.h5`, with
# `input_cell_types.h5`, `intermediate_cell_types.h5`, `output_cell_types.h5`) and the neuropil labels of its network
# figure (`flyvis/analysis/visualization/network_fig.py`, `NetworkFigure.add_neuropil_labels`):
#   layout "retina"        -> label "retina"
#   layout "intermediate"  -> label "lamina, medulla intrinsic cells, CT1"
#   layout "output"        -> label "T-shaped, transmedullary cells"
# The family inside a group follows the type name as expanded by that label and by the connectome nomenclature
# (L = lamina, Lawf = lamina wide-field, Mi = medulla intrinsic, CT1 as named, Tm and TmY = transmedullary,
# T1 to T5 = T-shaped). The three intermediate types that the label does not place (Am, C2, C3) are not guessed:
# they stay in "Other". Order inside a column is the order of `layout.h5`. Every one of the 65 cell types appears once.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # layout "retina" (= input_cell_types.h5)
    ("Retina", ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")),
    # layout "intermediate", family "lamina" (L1-L5 lamina monopolar, Lawf1-2 lamina wide-field)
    ("Lamina", ("L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2")),
    # layout "intermediate", family "medulla intrinsic cells" (Mi)
    ("Medulla intrinsic", ("Mi1", "Mi2", "Mi3", "Mi4", "Mi9", "Mi10", "Mi11", "Mi12", "Mi13", "Mi14", "Mi15")),
    # layout "intermediate", named "CT1" in the flyvis label
    ("CT1", ("CT1(Lo1)", "CT1(M10)")),
    # layout "output", family "T-shaped" (T1, T2, T2a, T3, T4a-d, T5a-d)
    ("T-shaped", ("T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")),
    # layout "output", family "transmedullary" (Tm and TmY)
    ("Transmedullary", ("Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20", "Tm28",
                        "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18")),
    # layout "intermediate", but the flyvis label does not say which family: left unplaced
    ("Other", ("Am", "C2", "C3")),
)
# Types that get a label box in the full figure (the rest are dots; the preview has none).
KEY_LABELS = frozenset({"R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "Mi1", "Tm3", "T4a", "T5a"})

TOP = 61  # the same top 61 as everywhere (10% of the 604 classes)
SOCIAL_PX = (1280, 640)
TITLE = "flyvis-circuit-bench"
COLOUR_PERCENTILES = (2, 98)  # the colour scale is clipped here; effects span many orders of magnitude
# cividis in the direction that makes a large effect the high-contrast end against the background:
CMAP = {"light": "cividis_r", "dark": "cividis"}


def positions() -> dict[str, tuple[float, float]]:
    """Node centre per cell type: column = layer; nodes are centred in the column with the same pitch as the
    8 photoreceptors at most (1/7 of the height), so a column with few types stays compact."""
    pos = {}
    for col, (_, types) in enumerate(LAYERS):
        n = len(types)
        step = min(1.0 / (n - 1), 1.0 / 7.0) if n > 1 else 0.0
        for i, t in enumerate(types):
            pos[t] = (float(col), 0.5 + ((n - 1) / 2 - i) * step)
    return pos


def build_table(data: dict) -> dict:
    """Per class: source, target, synapse count, mean effect over the models present, top-61 flag."""
    from .v11.z1 import truth_top_k

    models = [m for m in MODELS_POOLED if "B_C1" in data["models"].get(m, {})]
    if not models:
        raise ValueError("no ablation file in the package")
    first = data["models"][models[0]]["B_C1"]
    e = np.mean([data["models"][m]["B_C1"]["E_B"] for m in models], axis=0).astype(float)
    known = {t for _, ts in LAYERS for t in ts}
    src, dst = [str(x) for x in first["zrodlo"]], [str(x) for x in first["cel"]]
    unknown = sorted((set(src) | set(dst)) - known)
    if unknown:
        raise ValueError(f"cell types without a layer: {unknown}")
    per = [np.asarray(data["models"][m]["B_C1"]["E_B"], dtype=float) for m in models]
    n_top = np.sum([truth_top_k(x, TOP) for x in per], axis=0).astype(int)
    low_everywhere = np.all([x < np.median(x) for x in per], axis=0)
    return {"source": src, "target": dst, "synapses": np.asarray(first["S"], dtype=float), "effect": e,
            "top": truth_top_k(e, TOP), "n_models": len(models), "n_top": n_top, "low_everywhere": low_everywhere}


def line_width(synapses: np.ndarray, smax: float) -> np.ndarray:
    """Width in points, proportional to the square root of the synapse count (so the area of a bundle follows it)."""
    return 0.3 + 4.2 * np.sqrt(np.asarray(synapses, dtype=float) / smax)


def bezier(p0, c0, c1, p1, n: int = 36) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    a, b, c, d = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3)
    return a * np.array(p0) + b * np.array(c0) + c * np.array(c1) + d * np.array(p1)


def edge_curve(p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    """Between columns an S curve; inside a column an arc to the right (downward edges) or left (upward);
    a self connection is a small loop on the right of the node."""
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1 and y0 == y1:
        return bezier(p0, (x0 + 0.22, y0 + 0.035), (x0 + 0.22, y0 - 0.035), p1)
    if x0 == x1:
        side = 1.0 if y1 < y0 else -1.0
        bulge = side * (0.14 + 0.5 * abs(y1 - y0))
        return bezier(p0, (x0 + bulge, y0), (x0 + bulge, y1), p1)
    mid = 0.5 * (x0 + x1)
    return bezier(p0, (mid, y0), (mid, y1), p1)


def _norm(effect: np.ndarray):
    from matplotlib.colors import LogNorm

    pos = effect[effect > 0]
    lo, hi = np.percentile(pos, COLOUR_PERCENTILES)
    return LogNorm(vmin=float(lo), vmax=float(hi), clip=True)


def draw_graph(ax, table: dict, theme: dict, theme_name: str, labels: frozenset | set = frozenset(),
               node_pt: float = 8.5, head_pt: float = 11.0, wrap_titles: bool = False):
    """Edges (low effect first, the top 61 last with a casing), then the nodes; returns the colour mapping."""
    import matplotlib as mpl
    from matplotlib.collections import LineCollection

    pos = positions()
    smax = float(table["synapses"].max())
    norm = _norm(table["effect"])
    cmap = mpl.colormaps[CMAP[theme_name]]
    colours = cmap(norm(np.where(table["effect"] > 0, table["effect"], norm.vmin)))
    order = np.lexsort((np.arange(len(table["effect"])), table["effect"]))  # ascending, ties by class index
    lw = line_width(table["synapses"], smax)
    curves = [edge_curve(pos[s], pos[t]) for s, t in zip(table["source"], table["target"], strict=True)]
    for sel, alpha, casing in ((~table["top"], 0.34, False), (table["top"], 0.96, True)):
        idx = [i for i in order if sel[i]]
        rgba = colours[idx].copy()
        rgba[:, 3] = alpha
        if casing:
            ax.add_collection(LineCollection([curves[i] for i in idx], colors=theme["bg"],
                                             linewidths=lw[idx] + 1.7, capstyle="round", zorder=3))
        ax.add_collection(LineCollection([curves[i] for i in idx], colors=rgba, linewidths=lw[idx],
                                         capstyle="round", zorder=4 if casing else 2))
    for col, (title, types) in enumerate(LAYERS):
        ax.text(col, 1.075, title.replace(" ", "\n") if wrap_titles else title, ha="center", va="bottom",
                fontsize=head_pt, fontweight="bold", color=theme["fg"], zorder=8, linespacing=1.1)
        for t in types:
            x, y = pos[t]
            if t in labels:
                ax.text(x, y, t, ha="center", va="center", fontsize=node_pt, color=theme["fg"], zorder=7,
                        bbox={"boxstyle": "round,pad=0.22", "fc": theme["bg"], "ec": theme["fg"], "lw": 0.7})
            else:
                ax.scatter([x], [y], s=26, facecolor=theme["bg"], edgecolor=theme["fg"], linewidth=0.9, zorder=7)
    ax.set_xlim(-0.55, len(LAYERS) - 1 + 0.75)
    ax.set_ylim(-0.08, 1.16)
    ax.axis("off")
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _width_legend(ax, theme: dict, smax: float, fontsize: float, loc: str, anchor) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=theme["muted"], lw=float(line_width(np.array([s]), smax)[0]),
                      label=f"{s:,} synapses".replace(",", " ")) for s in (1_000, 10_000, 100_000)]
    handles.append(Line2D([], [], color=theme["fg"], lw=0.7, marker="o", mfc=theme["bg"], mec=theme["fg"], ms=0,
                          label="outlined: top 61 by ablation effect"))
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, fontsize=fontsize, frameon=False,
                    handlelength=3.0, labelcolor=theme["fg"])
    ax.add_artist(leg)


def _colorbar(fig, mappable, rect, theme: dict, fontsize: float, label: str):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.solids.set_rasterized(False)  # vector colour bar: no embedded base64 image in the SVG
    cb.outline.set_edgecolor(theme["muted"])
    cb.ax.tick_params(labelsize=fontsize, colors=theme["fg"])
    cb.set_label(label, fontsize=fontsize, color=theme["fg"], loc="right")
    return cb


PNG_DPI = 150
PNG_COLOURS = 256


def _median_cut(rgb: np.ndarray, n_colors: int, keep_background: bool) -> tuple[np.ndarray, np.ndarray]:
    """(palette (m, 3) uint8, palette index per pixel) of pixels (n, 3) by median cut on the distinct colours.
    The most frequent colour is kept exactly as entry 0 when `keep_background`."""
    keys = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(float)
    weight = counts**0.25  # rare colours keep palette entries next to the many frequent ones
    box_of = np.zeros(len(uniq), dtype=np.int64)
    first = 0
    if keep_background:
        background = int(np.argmax(counts))
        rest = np.array([i for i in range(len(uniq)) if i != background], dtype=np.int64)
        first = 1
    else:
        rest = np.arange(len(uniq), dtype=np.int64)
    boxes = [rest] if rest.size else []
    while len(boxes) < n_colors - first:
        splittable = [(weight[b].sum() * np.ptp(cols[b], axis=0).max(), i) for i, b in enumerate(boxes) if len(b) > 1]
        if not splittable:
            break
        _, i = max(splittable)
        b = boxes.pop(i)
        axis = int(np.argmax(np.ptp(cols[b], axis=0)))
        order = b[np.argsort(cols[b, axis], kind="stable")]
        cum = np.cumsum(weight[order])
        cut = min(max(int(np.searchsorted(cum, cum[-1] / 2)), 0), len(order) - 2) + 1
        boxes += [order[:cut], order[cut:]]
    palette = [cols[background]] if keep_background else []
    for k, b in enumerate(boxes, start=first):
        palette.append((cols[b] * counts[b][:, None]).sum(axis=0) / counts[b].sum())
        box_of[b] = k
    return np.rint(np.array(palette)).astype(np.uint8), box_of[inverse]


def quantized_png(rgba: np.ndarray, regions=(), region_colours: int = 64, n_colors: int = PNG_COLOURS) -> bytes:
    """An 8-bit palette PNG of an opaque RGBA image (height, width, 4): median cut on the distinct colours, the
    background kept exactly, a palette of its own for every `region` (x0, y0, x1, y1 in pixels, y from the top, e.g.
    a colour bar, so that its gradient does not band), Sub-filtered rows, zlib level 9. Deterministic (no timestamps,
    no metadata)."""
    import struct
    import zlib

    h, w = rgba.shape[:2]
    rgb = np.ascontiguousarray(rgba[..., :3])
    inside = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in regions:
        inside[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
    n_main = n_colors - region_colours * len(regions)
    pal, main_idx = _median_cut(rgb[~inside], n_main, keep_background=True)
    idx = np.zeros((h, w), dtype=np.int64)
    idx[~inside] = main_idx
    palette = [pal]
    offset = len(pal)
    for x0, y0, x1, y1 in regions:
        box = np.zeros((h, w), dtype=bool)
        box[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
        rp, ri = _median_cut(rgb[box], region_colours, keep_background=False)
        idx[box] = ri + offset
        palette.append(rp)
        offset += len(rp)
    pal = np.concatenate(palette)
    idx = idx.astype(np.uint8)
    filtered = idx.copy()
    filtered[:, 1:] = (idx[:, 1:].astype(np.int16) - idx[:, :-1].astype(np.int16)).astype(np.uint8)
    raw = np.concatenate([np.ones((h, 1), dtype=np.uint8), filtered], axis=1).tobytes()  # filter type 1 (Sub)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
            + chunk(b"PLTE", pal.tobytes()) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _save(fig, out: Path, stem: str, theme_name: str) -> list[Path]:
    """The SVG as drawn (tight box); the PNG as a quantized 8-bit palette image of the canvas (< 600 KB). The axes in
    `fig.png_regions` (the colour bar) get a palette of their own."""
    svg = out / f"{stem}-{theme_name}.svg"
    fig.savefig(svg, format="svg", dpi=200, bbox_inches="tight", pad_inches=0.15,
                metadata={"Date": None, "Creator": None})
    fig.set_dpi(PNG_DPI)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    height = rgba.shape[0]
    regions = []
    for axes in getattr(fig, "png_regions", ()):
        bb = axes.get_window_extent()
        regions.append((int(np.floor(bb.x0)), int(height - np.ceil(bb.y1)), int(np.ceil(bb.x1)),
                        int(height - np.floor(bb.y0))))
    png = out / f"{stem}-{theme_name}.png"
    png.write_bytes(quantized_png(rgba, regions))
    return [png, svg]


def connectome_figure(table: dict, theme_name: str):
    theme = THEMES[theme_name]
    plt = _mpl(theme)
    fig = plt.figure(figsize=(11.0, 8.6))
    ax = fig.add_axes((0.015, 0.17, 0.97, 0.76))
    mappable = draw_graph(ax, table, theme, theme_name, labels=KEY_LABELS)
    fig.text(0.015, 0.985, "Thick is not the same as important", fontsize=17, fontweight="bold", color=theme["fg"],
             va="top", ha="left")
    _width_legend(ax, theme, float(table["synapses"].max()), 11, "lower left", (0.0, -0.17))
    cb = _colorbar(fig, mappable, (0.58, 0.075, 0.38, 0.022), theme, 11,
                   "mean ablation effect, 10 models (log scale, clipped at the 2nd and 98th percentile)")
    fig.png_regions = [cb.ax]  # the gradient gets a palette of its own in the quantized PNG
    return fig, plt


# Social preview: everything pale grey and thin, two groups picked out in two colours (no colour scale).
N_HIGHLIGHT = 3
THICK_OUTSIDE = 9  # outside the top 61 in at least this many models
THIN_INSIDE = 5  # inside the top 61 in at least this many models
THICK_COLOUR = "#8B6B4A"  # warm grey-brown: many synapses, little effect
CRITICAL_COLOUR = "#0072B2"  # Okabe-Ito blue: few synapses, large effect
PALE_GREY = "#dde1e6"
HEADLINE = "Thick is not\nthe same as\nimportant"
PREVIEW_COLUMN_TITLES = ("Retina", "Other")  # first and last column: the direction of the signal
CAPTION = "thickness = synapse count \u00b7 color = effect when removed"


def highlight_classes(table: dict, n: int = N_HIGHLIGHT) -> dict[str, list[int]]:
    """The classes the preview picks out, by a fixed rule (ties by class index), stable across the models:

    * thick, little effect: the n classes with the most synapses among those outside the top 61 in at least
      `THICK_OUTSIDE` of the models (9 of 10);
    * thin, critical: the n classes with the fewest synapses among those in the top 61 in at least `THIN_INSIDE`
      of the models (5 of 10).
    """
    s = table["synapses"]
    thick = np.flatnonzero(table["n_models"] - table["n_top"] >= THICK_OUTSIDE)
    thin = np.flatnonzero(table["n_top"] >= THIN_INSIDE)
    return {"thick": [int(i) for i in thick[np.lexsort((thick, -s[thick]))][:n]],
            "thin": [int(i) for i in thin[np.lexsort((thin, s[thin]))][:n]]}


def synapse_hits(results: dict) -> int:
    """How many of the 61 important connections the top 61 by synapse count contains (mean over the 10 models,
    precision@61 of the synapse count from the Z1 table of results.json)."""
    try:
        row = next(x for x in results["v1_1"]["z1"]["table"] if x["method"] == "S" and x["k"] == TOP)
        return round(row["precision_at_k"]["pooled"]["mean"] * TOP)
    except (KeyError, StopIteration) as e:
        raise ValueError("the social preview needs results.json with v1_1.z1 (run fcb reproduce)") from e


def preview_line_width(synapses, smax: float):
    return 0.8 + 5.2 * np.sqrt(np.asarray(synapses, dtype=float) / smax)


def group_widths(table: dict, sel: dict[str, list[int]]) -> dict[str, float]:
    """Mean drawn width (points) of each picked group; the legend swatches use exactly these."""
    smax = float(table["synapses"].max())
    return {k: float(np.mean(preview_line_width(table["synapses"][v], smax))) for k, v in sel.items()}


def draw_preview_graph(ax, table: dict, sel: dict[str, list[int]]) -> None:
    from matplotlib.collections import LineCollection

    pos = positions()
    smax = float(table["synapses"].max())
    curves = [edge_curve(pos[s], pos[t]) for s, t in zip(table["source"], table["target"], strict=True)]
    picked = {i for v in sel.values() for i in v}
    rest = [i for i in range(len(curves)) if i not in picked]
    ax.add_collection(LineCollection([curves[i] for i in rest], colors=PALE_GREY, linewidths=0.45, zorder=2))
    for key, colour in (("thick", THICK_COLOUR), ("thin", CRITICAL_COLOUR)):
        idx = sel[key]
        ax.add_collection(LineCollection([curves[i] for i in idx], colors=colour, capstyle="round", zorder=5,
                                         linewidths=preview_line_width(table["synapses"][idx], smax)))
    ends = {c: {n for i in sel[k] for n in (table["source"][i], table["target"][i])}
            for k, c in (("thick", THICK_COLOUR), ("thin", CRITICAL_COLOUR))}
    for col, (title, types) in enumerate(LAYERS):
        if title in PREVIEW_COLUMN_TITLES:
            ax.text(col, 1.075, title, ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="#6b7280", zorder=8)
        for ty in types:
            x, y = pos[ty]
            hit = [c for c, names in ends.items() if ty in names]
            ax.scatter([x], [y], s=34 if hit else 18, facecolor=hit[0] if hit else "#ffffff",
                       edgecolor=hit[0] if hit else "#b8bec6", linewidth=0.8, zorder=7)
    ax.set_xlim(-0.55, len(LAYERS) - 1 + 0.75)
    ax.set_ylim(-0.08, 1.16)
    ax.axis("off")


def social_preview(table: dict, results: dict):
    """1280 x 640 px, light theme: headline, one number, two colour categories; content at least 40 px from the
    left and right edge (LinkedIn crops to 1.91:1)."""
    import textwrap

    from matplotlib.lines import Line2D

    theme = THEMES["light"]
    plt = _mpl(theme)
    w, h = SOCIAL_PX
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0.43, 0.11, 0.535, 0.80))
    sel = highlight_classes(table)
    draw_preview_graph(ax, table, sel)
    gw = group_widths(table, sel)
    x0 = 60 / w
    fig.text(x0, 0.94, TITLE, fontsize=19, color="#6b7280", va="top", ha="left")
    fig.text(x0, 0.87, HEADLINE, fontsize=38, fontweight="bold", color=theme["fg"], va="top", ha="left",
             linespacing=1.1)
    n = synapse_hits(results)
    fig.text(x0, 0.49, "\n".join(textwrap.wrap(f"Synapse count finds {n} of the {TOP} connections that matter", 30)),
             fontsize=19.5, fontweight="bold", color=theme["fg"], va="top", ha="left", linespacing=1.3)
    for y, colour, label, key in ((0.255, THICK_COLOUR, "thick, little effect", "thick"),
                                  (0.185, CRITICAL_COLOUR, "thin, critical", "thin")):
        fig.add_artist(Line2D([x0, x0 + 0.045], [y, y], transform=fig.transFigure, color=colour, lw=gw[key],
                              solid_capstyle="round"))
        fig.text(x0 + 0.06, y, label, fontsize=17, color=theme["fg"], va="center", ha="left")
    fig.text(x0, 0.035, CAPTION, fontsize=13.5, color="#6b7280", va="bottom", ha="left")
    return fig, plt


def make_figures(data: dict, out: Path, results: dict) -> list[Path]:
    """connectome-{light,dark}.{svg,png} and social-preview.png from the loaded measurement package; the preview
    also needs results.json (the number of important connections the synapse count finds)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = build_table(data)
    paths: list[Path] = []
    for theme_name in THEMES:
        fig, plt = connectome_figure(table, theme_name)
        paths += _save(fig, out, "connectome", theme_name)
        plt.close(fig)
    fig, plt = social_preview(table, results)
    p = out / "social-preview.png"
    fig.savefig(p, format="png", dpi=100, metadata={"Software": None})  # exact pixel size: no tight bbox
    plt.close(fig)
    return paths + [p]

