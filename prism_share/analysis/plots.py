"""Thesis figures: vector PDF (and SVG for Markdown), sized for an A4 text block.

Figures consume the tidy summary tables produced by the pilot (and, later, by
sweep.py): one row per (colour_depth, cell_px, series, severity, decoder) with
columns shape_ser, colour_ser, goodput_mbps.

Encoding conventions, identical in every figure:
* colour depth is ordered, so depths 2..16 share one blue ramp, light -> dark;
* monochrome (depth 1), the hypothesis' candidate, is the single orange accent;
* every series also has its own marker, so identity never rests on colour.
The palette was checked with the dataviz validator (ordinal ramp: monotone,
single hue; accent vs blue: CVD dE 24.7).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from prism_share.codec.params import ALLOWED_COLOUR_DEPTHS  # noqa: E402
from prism_share.sim.degrade import SEVERITY_UNITS  # noqa: E402

TEXT_WIDTH_IN = 6.3  # 160 mm
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
#: Type sizes in points.
PT_BODY, PT_TITLE, PT_SMALL, PT_CELL = 7.5, 8.5, 6.5, 6.0
MARKER_PT = 3.5
#: Monochrome = orange accent; colour depths in ascending order = blue ramp light -> dark.
_ACCENT = "#eb6834"
_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_MARKERS = ["o", "s", "^", "D", "v"]
DEPTH_STYLE: dict[int, tuple[str, str]] = {
    depth: (_ACCENT if depth == 1 else _RAMP[i - 1], _MARKERS[i]) for i, depth in enumerate(ALLOWED_COLOUR_DEPTHS)
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
SERIES_TITLES = {
    "clean": "undegraded",
    "blur": "Blur",
    "noise": "Sensor noise",
    "perspective": "Perspective warp",
    "white_balance": "White-balance shift",
    "chroma_nearest": "Chroma 4:2:0, nearest",
    "chroma_bilinear": "Chroma 4:2:0, bilinear",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": PT_BODY,
            "axes.titlesize": PT_TITLE,
            "axes.labelsize": PT_BODY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.facecolor": SURFACE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", metadata={"Date": None} if ext == "svg" else {"CreationDate": None})
    plt.close(fig)


def _series_frame(summary: pd.DataFrame, series: str, decoder: str) -> pd.DataFrame:
    """Rows of one series plus the undegraded rows as severity 0 (not for chroma, whose identity is pitch 1)."""
    s = summary[(summary["decoder"] == decoder)]
    own = s[s["series"] == series]
    if series.startswith("chroma"):
        return own
    return pd.concat([s[s["series"] == "clean"], own])


def _depth_legend(fig: plt.Figure, depths: Sequence[int]) -> None:
    handles = [
        plt.Line2D([], [], color=DEPTH_STYLE[d][0], marker=DEPTH_STYLE[d][1], markersize=MARKER_PT + 0.5, linewidth=1.5,
                   label="monochrome" if d == 1 else f"{d} colours")
        for d in depths
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02), fontsize=PT_BODY, handlelength=2.2)


def best_goodput_vs_severity(summary: pd.DataFrame, decoder: str, series: Sequence[str], out_dir: Path, name: str) -> None:
    """Small multiples: max-over-cell-size goodput vs severity, one line per colour depth."""
    _style()
    cols = 3
    rows = -(-len(series) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(TEXT_WIDTH_IN, 2.0 * rows + 0.3), squeeze=False)
    depths = sorted(summary["colour_depth"].unique())
    for ax, name_ in zip(axes.flat, series, strict=False):
        df = _series_frame(summary, name_, decoder)
        best = df.groupby(["colour_depth", "severity"])["goodput_mbps"].max().reset_index()
        for d in depths:
            b = best[best["colour_depth"] == d].sort_values("severity")
            colour, marker = DEPTH_STYLE[int(d)]
            ax.plot(b["severity"], b["goodput_mbps"], color=colour, marker=marker, markersize=MARKER_PT, linewidth=1.5,
                    zorder=3 if d == 1 else 2)
        ax.set_title(SERIES_TITLES.get(name_, name_), loc="left")
        degradation = df[df["series"] == name_]["degradation"].iloc[0]
        ax.set_xlabel(SEVERITY_UNITS[degradation].split(";")[0], fontsize=PT_SMALL)
        ax.set_ylim(bottom=0)
        ax.tick_params(length=2)
    for ax in axes.flat[len(series):]:
        ax.set_visible(False)
    for r in range(rows):
        axes[r, 0].set_ylabel("best goodput (Mbit/s)")
    _depth_legend(fig, [int(d) for d in depths])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, name)


def error_decomposition(summary: pd.DataFrame, decoder: str, series: Sequence[str], cell_px: int, out_dir: Path, name: str) -> None:
    """Shape SER (top row) and colour SER (bottom row) vs severity at one cell size, per colour depth."""
    _style()
    fig, axes = plt.subplots(2, len(series), figsize=(TEXT_WIDTH_IN, 3.6), squeeze=False)
    depths = sorted(summary["colour_depth"].unique())
    for j, name_ in enumerate(series):
        df = _series_frame(summary, name_, decoder)
        df = df[df["cell_px"] == cell_px]
        for i, metric in enumerate(["shape_ser", "colour_ser"]):
            ax = axes[i, j]
            for d in depths:
                b = df[df["colour_depth"] == d].sort_values("severity")
                colour, marker = DEPTH_STYLE[int(d)]
                ax.plot(b["severity"], 100 * b[metric], color=colour, marker=marker, markersize=MARKER_PT - 0.5, linewidth=1.3,
                        zorder=3 if d == 1 else 2)
            # An all-zero panel gets a 0-1 % axis rather than matplotlib's arbitrary tiny range.
            ax.set_ylim(0, max(1.0, 1.08 * 100 * float(df[metric].max())))
            ax.tick_params(length=2)
            if i == 0:
                ax.set_title(SERIES_TITLES.get(name_, name_), loc="left")
            else:
                degradation = df[df["series"] == name_]["degradation"].iloc[0]
                ax.set_xlabel(SEVERITY_UNITS[degradation].split(";")[0].split("(")[0].strip(), fontsize=PT_SMALL)
        axes[0, 0].set_ylabel("shape SER (%)")
        axes[1, 0].set_ylabel("colour SER (%)")
    _depth_legend(fig, [int(d) for d in depths])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, name)


def goodput_surface(summary: pd.DataFrame, decoder: str, panels: Sequence[tuple[str, float, str]], out_dir: Path, name: str) -> None:
    """Heatmaps of goodput over colour depth x cell size; the maximum cell is outlined."""
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH_IN, 2.5), squeeze=False)
    s = summary[summary["decoder"] == decoder]
    grids = []
    for series, severity, _ in panels:
        g = s[(s["series"] == series) & (s["severity"] == severity)].pivot(index="colour_depth", columns="cell_px", values="goodput_mbps")
        grids.append(g)
    vmax = max(float(g.values.max()) for g in grids)
    image = None
    for ax, g, (_, _, title) in zip(axes.flat, grids, panels, strict=True):
        values = g.values
        image = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        ax.grid(False)
        ax.set_xticks(range(len(g.columns)), [str(c) for c in g.columns])
        ax.set_yticks(range(len(g.index)), [str(d) for d in g.index])
        ax.set_xlabel("cell size (px)")
        ax.set_title(title, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        r, c = np.unravel_index(np.argmax(values), values.shape)
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
        for (ri, ci), v in np.ndenumerate(values):
            ax.text(ci, ri, f"{v:.1f}", ha="center", va="center", fontsize=PT_CELL,
                    color="#ffffff" if v > 0.55 * vmax else INK, fontweight="bold" if (ri, ci) == (r, c) else "normal")
    axes[0, 0].set_ylabel("colour depth")
    assert image is not None
    bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    bar.set_label("goodput (Mbit/s)")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    _save(fig, out_dir, name)


def pilot_figures(summary: pd.DataFrame, decoder: str, out_dir: Path, decomposition_cell_px: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_goodput_vs_severity(
        summary, decoder,
        ["blur", "noise", "perspective", "white_balance", "chroma_nearest", "chroma_bilinear"],
        out_dir, "fig_best_goodput",
    )
    error_decomposition(summary, decoder, ["blur", "noise", "chroma_nearest", "chroma_bilinear"], decomposition_cell_px, out_dir,
        "fig_error_decomposition",
    )
    goodput_surface(
        summary, decoder,
        [("chroma_nearest", 2.0, "4:2:0, nearest"), ("chroma_bilinear", 2.0, "4:2:0, bilinear"), ("blur", 1.0, "blur σ = 1 px")],
        out_dir, "fig_goodput_surface",
    )
