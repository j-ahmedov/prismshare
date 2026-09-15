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

from prism_share.codec.decoder import HEADLINE_DECODER  # noqa: E402
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


# =========================================================================== #
# Figures from captured data. Every one carries its provenance on its face:
# run ids, device model, the analysis choices, and the number of frames behind
# each point. A figure without that cannot be defended.
# =========================================================================== #

#: Three-outcome yield colours: status semantics (good / serious / critical), always with labels.
OUTCOME_STYLE: dict[str, tuple[str, str]] = {
    "recovered": ("#0ca30c", "recovered"),
    "not_recoverable": ("#ec835a", "detected, not recoverable"),
    "detection_failed": ("#d03b3b", "detection failed"),
}


def _condition_columns(summary: pd.DataFrame) -> list[str]:
    return sorted(c for c in summary.columns if c.startswith("condition_"))


def decoder_role(label: str) -> str:
    """A decoder label with its status: the pre-registered headline or a sensitivity variant."""
    return f"{label} (pre-registered headline)" if label == HEADLINE_DECODER.label else f"{label} (sensitivity)"


def _selection_text(rows: pd.DataFrame) -> str:
    modes = set(rows["rs_selection"].unique()) if "rs_selection" in rows else {"unknown"}
    if modes == {"out_of_sample"}:
        return "chosen out of sample"
    return "IN-SAMPLE RS SELECTION (optimistic, not a result)" if "in_sample" in modes else "RS selection unknown"


def provenance_text(rows: pd.DataFrame, rs: str) -> str:
    """One provenance block for the rows a figure is drawn from."""
    runs = sorted(rows["run_id"].unique())
    devices = sorted({f"{m} {d}" for m, d in zip(rows["device_manufacturer"], rows["device_model"], strict=True)})
    yuv = sorted({f"{m} {r} ({s})" for m, r, s in zip(rows["yuv_matrix"], rows["yuv_range"], rows["yuv_range_source"], strict=True)})
    flat = "on" if rows["flat_field"].any() else "off"
    lines = [
        f"CAPTURED DATA  ·  device: {', '.join(devices)}  ·  pixel source: {', '.join(sorted(rows['pixel_source'].unique()))}"
        f"  ·  decoder: {', '.join(decoder_role(d) for d in sorted(rows['decoder'].unique()))}  ·  RS: {rs}, {_selection_text(rows)}  ·  kernel: {', '.join(sorted(rows['kernel'].unique()))}"
        f"  ·  flat field: {flat}  ·  YUV: {', '.join(yuv)}  ·  n = code frames behind each point",
        f"runs ({len(runs)}): {', '.join(runs)}",
    ]
    return "\n".join(lines)


_FOOTER_PT = PT_CELL - 1


def _stamp(fig: plt.Figure, rows: pd.DataFrame, rs: str) -> None:
    """Provenance as the figure's bottom label, so constrained layout reserves room for it."""
    import textwrap

    chars = int(fig.get_figwidth() * 72 / (_FOOTER_PT * 0.62))  # monospace advance ~0.6 em
    text = "\n".join(textwrap.fill(line, chars) for line in provenance_text(rows, rs).splitlines())
    fig.supxlabel(text, x=0.0, ha="left", fontsize=_FOOTER_PT, color=INK_2, family="monospace")


def _groups(summary: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split by device and condition: one figure panel set per physical situation."""
    keys = ["device_model", *_condition_columns(summary)]
    out = []
    for key, g in summary.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        label = ", ".join(f"{k.replace('condition_', '')}={v}" for k, v in zip(keys, key, strict=True))
        out.append((label, g))
    return out


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:120]


GOODPUT_VIEWS: tuple[tuple[str, str], ...] = (
    ("goodput_mbps", "as measured (index band present)"),
    ("goodput_band_credited_mbps", "index band's cells credited back"),
)


def captured_goodput_surface(summary: pd.DataFrame, out_dir: Path, *, rs: str = "best") -> list[Path]:
    """Goodput over colour depth x cell size, per device/condition: measured and band-credited side by side.

    Both panels share one colour scale. Every cell shows goodput and n; the
    band-credited panel also shows the band's cell cost for that configuration,
    so the reader can see that the apparatus is a small, configuration-dependent
    term and check whether it moves the maximum.
    """
    _style()
    paths = []
    for label, g in _groups(summary):
        counts = g.pivot_table(index="colour_depth", columns="cell_px", values="n_code_frames", aggfunc="sum")
        cost = g.pivot_table(index="colour_depth", columns="cell_px", values="band_cell_cost_pct", aggfunc="mean")
        pivots = [g.pivot_table(index="colour_depth", columns="cell_px", values=f"{rs}_{col}", aggfunc="mean")
                  for col, _ in GOODPUT_VIEWS]
        top = max(float(np.nanmax(pv.to_numpy(dtype=float))) for pv in pivots) if len(g) else 1.0
        fig, axes = plt.subplots(1, len(GOODPUT_VIEWS), figsize=(TEXT_WIDTH_IN, 3.6), layout="constrained", squeeze=False)
        image = None
        for ax, pivot, (column, title) in zip(axes.flat, pivots, GOODPUT_VIEWS, strict=True):
            values = pivot.to_numpy(dtype=float)
            image = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=top, aspect="auto", origin="lower")
            ax.grid(False)
            ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(d) for d in pivot.index])
            ax.set_xlabel("cell size (px)")
            ax.set_title(title, loc="left", fontsize=PT_BODY)
            best = np.unravel_index(np.nanargmax(values), values.shape) if np.isfinite(values).any() else None
            for (r, c), v in np.ndenumerate(values):
                n = counts.to_numpy()[r, c]
                extra = f"\nband {cost.to_numpy()[r, c]:.1f} %" if column != "goodput_mbps" and np.isfinite(v) else ""
                text = "–" if not np.isfinite(v) else f"{v:.2f}\nn={int(n)}{extra}"
                ax.text(c, r, text, ha="center", va="center", fontsize=PT_CELL - 1,
                        color="#ffffff" if np.isfinite(v) and v > 0.55 * top else INK,
                        fontweight="bold" if best is not None and (r, c) == best else "normal")
            if best is not None:
                ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
        axes[0, 0].set_ylabel("colour depth")
        fig.suptitle(f"Goodput, {label}  (outlined = maximum in each panel)", x=0.0, ha="left", fontsize=PT_TITLE)
        assert image is not None
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
        bar.set_label("goodput (Mbit/s)")
        bar.outline.set_visible(False)
        _stamp(fig, g, rs)
        name = f"captured_goodput_{_slug(label)}"
        _save(fig, out_dir, name)
        paths.append(out_dir / f"{name}.pdf")
    return paths


def captured_outcomes(summary: pd.DataFrame, out_dir: Path, *, rs: str = "best") -> list[Path]:
    """Three-outcome yield per configuration: detection failed / detected not recoverable / recovered."""
    _style()
    paths = []
    for label, g in _groups(summary):
        g = g.sort_values(["colour_depth", "cell_px"])
        names = [f"{int(d)} col · {int(c)} px" for d, c in zip(g["colour_depth"], g["cell_px"], strict=True)]
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 1.6 + 0.24 * len(g)), layout="constrained")
        left = np.zeros(len(g))
        n = g["n_code_frames"].to_numpy(dtype=float)
        for key in ("recovered", "not_recoverable", "detection_failed"):
            colour, text = OUTCOME_STYLE[key]
            frac = np.divide(g[f"{rs}_n_{key}"].to_numpy(dtype=float), n, out=np.zeros(len(g)), where=n > 0)
            ax.barh(range(len(g)), frac, left=left, color=colour, edgecolor=SURFACE, linewidth=1.0, label=text, height=0.7)
            left += frac
        for i, count in enumerate(n):
            ax.text(1.01, i, f"n={int(count)}", va="center", fontsize=PT_CELL, color=INK_2)
        ax.set_yticks(range(len(g)), names)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of code frames")
        ax.set_title(f"Frame outcomes, {label}", loc="left")
        ax.grid(axis="y", visible=False)
        fig.legend(loc="outside upper right", ncol=3, fontsize=PT_BODY, frameon=False)
        _stamp(fig, g, rs)
        name = f"captured_outcomes_{_slug(label)}"
        _save(fig, out_dir, name)
        paths.append(out_dir / f"{name}.pdf")
    return paths


def captured_error_decomposition(summary: pd.DataFrame, out_dir: Path, *, rs: str = "best") -> list[Path]:
    """Shape SER and colour SER per configuration, side by side, with the identified-frame count."""
    _style()
    paths = []
    for label, g in _groups(summary):
        g = g.sort_values(["colour_depth", "cell_px"])
        names = [f"{int(d)}·{int(c)}" for d, c in zip(g["colour_depth"], g["cell_px"], strict=True)]
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.2), layout="constrained")
        x = np.arange(len(g))
        ax.plot(x, 100 * g["shape_ser"], linestyle="none", marker="o", color=_RAMP[2], markersize=MARKER_PT + 1, label="shape SER")
        ax.plot(x, 100 * g["colour_ser"], linestyle="none", marker="D", color=_ACCENT, markersize=MARKER_PT + 1, label="colour SER")
        for i, n in enumerate(g["n_identified"]):
            ax.text(i, 0, f"n={int(n)}", rotation=90, ha="center", va="bottom", fontsize=PT_CELL - 1, color=MUTED)
        ax.set_xticks(x, names, rotation=90)
        ax.set_xlabel("colour depth · cell size (px)")
        ax.set_ylabel("symbol error rate (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Error decomposition, {label}", loc="left")
        ax.legend(loc="upper left", fontsize=PT_BODY)
        _stamp(fig, g, rs)
        name = f"captured_errors_{_slug(label)}"
        _save(fig, out_dir, name)
        paths.append(out_dir / f"{name}.pdf")
    return paths


def device_comparison(summary: pd.DataFrame, out_dir: Path, *, rs: str = "best") -> list[Path]:
    """Best goodput (over cell sizes) per colour depth, one line per device; n at every point.

    Two panels, measured and with the index band's cells credited back, on one y scale.
    """
    _style()
    keys = _condition_columns(summary)
    paths = []
    groups = summary.groupby(keys, sort=True, dropna=False) if keys else [((), summary)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        label = ", ".join(f"{k.replace('condition_', '')}={v}" for k, v in zip(keys, key, strict=True)) or "all conditions"
        fig, axes = plt.subplots(1, len(GOODPUT_VIEWS), figsize=(TEXT_WIDTH_IN, 3.4), layout="constrained",
                                 squeeze=False, sharey=True)
        devices = sorted(g["device_model"].unique())
        styles = ["o", "s", "^", "D", "v"]
        top = max(float(np.nanmax(g[f"{rs}_{col}"])) for col, _ in GOODPUT_VIEWS) if len(g) else 1.0
        for ax, (column, title) in zip(axes.flat, GOODPUT_VIEWS, strict=True):
            for i, device in enumerate(devices):
                d = g[g["device_model"] == device]
                best = d.loc[d.groupby("colour_depth")[f"{rs}_{column}"].idxmax()].sort_values("colour_depth")
                colour = _RAMP[min(i, len(_RAMP) - 1)] if len(devices) > 1 else _RAMP[2]
                ax.plot(best["colour_depth"], best[f"{rs}_{column}"], marker=styles[i % len(styles)], color=colour,
                        linewidth=1.5, markersize=MARKER_PT + 0.5, label=device)
                for _, row in best.iterrows():
                    ax.annotate(f"n={int(row['n_code_frames'])}\n{int(row['cell_px'])} px", (row["colour_depth"], row[f"{rs}_{column}"]),
                                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=PT_CELL - 1, color=INK_2)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(g["colour_depth"].unique()), [str(int(v)) for v in sorted(g["colour_depth"].unique())])
            ax.set_xlabel("colour depth")
            ax.set_ylim(0, 1.25 * top if top > 0 else 1.0)  # headroom for the n / cell-size labels
            ax.set_title(title, loc="left", fontsize=PT_BODY)
        axes[0, 0].set_ylabel("best goodput (Mbit/s)")
        axes[0, -1].legend(fontsize=PT_BODY, loc="lower right")
        fig.suptitle(f"Per-device comparison, {label}", x=0.0, ha="left", fontsize=PT_TITLE)
        _stamp(fig, g, rs)
        name = f"captured_devices_{_slug(label)}"
        _save(fig, out_dir, name)
        paths.append(out_dir / f"{name}.pdf")
    return paths


def captured_figures(summary: pd.DataFrame, out_dir: Path, *, rs: str = "best") -> list[Path]:
    """All captured-data figures, one set per (pixel source, decoder) analysed.

    Figures from the pre-registered headline decoder go under ``headline/``,
    every other decoder's under ``sensitivity/``, so a sensitivity figure cannot
    be mistaken for a headline one by where it sits (its footer says so too).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (source, decoder), g in summary.groupby(["pixel_source", "decoder"], sort=True):
        role = "headline" if decoder == HEADLINE_DECODER.label else "sensitivity"
        sub = out_dir / role / _slug(f"{source}_{decoder}")
        sub.mkdir(parents=True, exist_ok=True)
        for fn in (captured_goodput_surface, captured_outcomes, captured_error_decomposition, device_comparison):
            paths += fn(g, sub, rs=rs)
    return paths
