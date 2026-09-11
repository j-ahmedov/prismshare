"""Pilot study: every code configuration x every degradation ladder, simulated.

    python -m prism_share.sim.pilot --config experiments/pilot.yaml

Pipeline per (configuration, condition, frame):

    encode -> degrade (one degradation, one severity) -> quantize to 8 bit
    -> decode with each decoder variant -> compare with ground truth

Recorded per decoded frame: shape SER, colour SER, byte error rate, and the
maximum per-codeword byte error count for each codeword length in
``ecc_lengths``. From those, ecc_sim derives frame yield and goodput for the
default RS(n, k) and for the goodput-maximising k.

Outputs:
* ``<data>/pilot_frames.parquet`` - one row per decoded frame (gitignored);
* ``<out>/pilot_summary.csv``     - one row per (config, condition, decoder);
* ``<out>/fig_*.pdf|svg``          - figures;
* ``<report>``                     - docs/pilot.md, regenerated; a block between
  ``<!-- interpretation -->`` and ``<!-- /interpretation -->`` is preserved.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from prism_share.analysis import ecc_sim
from prism_share.analysis.metrics import byte_errors, goodput_mbit_per_s, symbol_errors
from prism_share.codec.decoder import DecoderOptions, read_symbols
from prism_share.codec.encoder import encode_frames
from prism_share.codec.framing import frame_capacity
from prism_share.codec.params import ASSUMED_FPS, BITS_PER_BYTE, CodecParams
from prism_share.codec.prng import keystream
from prism_share.sim.degrade import SEVERITY_UNITS, Degradation, apply_chain, quantize

CLEAN = "clean"
MBIT = 1e6


@dataclass(frozen=True)
class Series:
    name: str
    degradation: str
    severities: tuple[float, ...]
    options: tuple[tuple[str, Any], ...] = ()

    def steps(self) -> list[Degradation]:
        return [Degradation(self.degradation, float(s), self.options) for s in self.severities]


@dataclass(frozen=True)
class PilotConfig:
    seed: int
    frames: int
    colour_depths: tuple[int, ...]
    cell_px: tuple[int, ...]
    ecc_lengths: tuple[int, ...]
    decoders: tuple[DecoderOptions, ...]
    headline_decoder: DecoderOptions
    series: tuple[Series, ...]

    def configs(self) -> list[CodecParams]:
        return [
            CodecParams(colour_depth=d, cell_px=c, seed=self.seed)
            for d, c in itertools.product(self.colour_depths, self.cell_px)
        ]


def load_config(path: str | Path) -> PilotConfig:
    doc = yaml.safe_load(Path(path).read_text())
    return PilotConfig(
        seed=int(doc["seed"]),
        frames=int(doc["frames_per_condition"]),
        colour_depths=tuple(int(d) for d in doc["colour_depths"]),
        cell_px=tuple(int(c) for c in doc["cell_px"]),
        ecc_lengths=tuple(int(n) for n in doc["ecc_lengths"]),
        decoders=tuple(DecoderOptions(**d) for d in doc["decoders"]),
        headline_decoder=DecoderOptions(**doc["headline_decoder"]),
        series=tuple(
            Series(
                name=s["name"],
                degradation=s["degradation"],
                severities=tuple(float(v) for v in s["severities"]),
                options=tuple(sorted((s.get("options") or {}).items())),
            )
            for s in doc["series"]
        ),
    )


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def _conditions(cfg: PilotConfig) -> list[tuple[str, Degradation | None]]:
    out: list[tuple[str, Degradation | None]] = [(CLEAN, None)]
    for series in cfg.series:
        out += [(series.name, step) for step in series.steps()]
    return out


def simulate_config(params: CodecParams, cfg: PilotConfig) -> list[dict[str, Any]]:
    """All conditions x frames x decoders for one configuration (runs in a worker)."""
    cap = frame_capacity(params)
    payload = keystream(params.seed, "pilot-payload", cap.block_bytes * cfg.frames)
    frames = encode_frames(payload, params, n_frames=cfg.frames)
    rows: list[dict[str, Any]] = []
    for series_name, step in _conditions(cfg):
        for index, frame in enumerate(frames):
            degraded = frame.image if step is None else quantize(apply_chain(frame.image, [step], params, index=index))
            for options in cfg.decoders:
                readout = read_symbols(degraded, params, options)
                errs = symbol_errors(frame.glyphs, frame.colours, readout.glyphs, readout.colours)
                berr = byte_errors(frame.glyphs, frame.colours, readout.glyphs, readout.colours, params)
                row = {
                    "colour_depth": params.colour_depth,
                    "cell_px": params.cell_px,
                    "series": series_name,
                    "degradation": step.name if step else CLEAN,
                    "severity": step.severity if step else 0.0,
                    "condition": step.label if step else CLEAN,
                    "decoder": options.label,
                    "frame": index,
                    "n_cells": errs.n_cells,
                    "shape_ser": errs.shape_ser,
                    "colour_ser": errs.colour_ser,
                    "symbol_ser": errs.symbol_ser,
                    "byte_error_rate": float(berr[: cap.stream_bytes].mean()),
                }
                for n in cfg.ecc_lengths:
                    row[f"max_cw_errors_n{n}"] = ecc_sim.max_codeword_errors(berr, n)
                rows.append(row)
    return rows


def run(cfg: PilotConfig, jobs: int) -> pd.DataFrame:
    configs = cfg.configs()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, result in enumerate(pool.map(simulate_config, configs, [cfg] * len(configs)), start=1):
            rows += result
            print(f"  [{i:2d}/{len(configs)}] {configs[i - 1].label}  ({time.monotonic() - started:.0f} s)", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def summarise(frames: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    """One row per (config, series, severity, decoder) with SERs, yields and goodputs."""
    keys = ["colour_depth", "cell_px", "series", "degradation", "severity", "condition", "decoder"]
    out = []
    for key, group in frames.groupby(keys, sort=True):
        record = dict(zip(keys, key, strict=True))
        params = CodecParams(colour_depth=int(record["colour_depth"]), cell_px=int(record["cell_px"]), seed=cfg.seed)
        record.update(
            frames=len(group),
            shape_ser=group["shape_ser"].mean(),
            colour_ser=group["colour_ser"].mean(),
            symbol_ser=group["symbol_ser"].mean(),
            byte_error_rate=group["byte_error_rate"].mean(),
        )
        fixed = ecc_sim.evaluate(group[f"max_cw_errors_n{params.ecc_total}"], params, params.ecc_total, params.ecc_data)
        record.update(
            fixed_rs=f"RS({fixed.n},{fixed.k})",
            fixed_yield=fixed.frame_yield,
            fixed_goodput_mbps=fixed.goodput_bytes_per_s * BITS_PER_BYTE / MBIT,
        )
        best = max(
            (ecc_sim.best_rate(group[f"max_cw_errors_n{n}"], params, n) for n in cfg.ecc_lengths),
            key=lambda o: (o.goodput_bytes_per_s, o.k / o.n),
        )
        record.update(
            best_rs=f"RS({best.n},{best.k})",
            best_code_rate=best.k / best.n,
            best_yield=best.frame_yield,
            best_payload_bytes=best.payload_bytes,
            goodput_mbps=best.goodput_bytes_per_s * BITS_PER_BYTE / MBIT,
            ceiling_mbps=goodput_mbit_per_s(frame_capacity(params).capacity_bytes, 1.0),
        )
        out.append(record)
    return pd.DataFrame(out)


def winners(summary: pd.DataFrame, decoder: str) -> pd.DataFrame:
    """Per (series, severity): overall best config, best monochrome, best colour."""
    s = summary[summary["decoder"] == decoder]
    rows = []
    for (series, severity), g in s.groupby(["series", "severity"], sort=False):
        g = g.sort_values(["goodput_mbps", "colour_depth", "cell_px"], ascending=[False, True, False])
        mono = g[g["colour_depth"] == 1].iloc[0]
        colour = g[g["colour_depth"] > 1].iloc[0]
        top = g.iloc[0]
        rows.append(
            {
                "series": series,
                "severity": severity,
                "winner": _cfg_name(top),
                "winner_mbps": top["goodput_mbps"],
                "winner_rs": top["best_rs"],
                "best_mono": _cfg_name(mono),
                "mono_mbps": mono["goodput_mbps"],
                "best_colour": _cfg_name(colour),
                "colour_mbps": colour["goodput_mbps"],
                "colour_over_mono": colour["goodput_mbps"] / mono["goodput_mbps"] if mono["goodput_mbps"] > 0 else np.inf,
                "colour_wins": bool(top["colour_depth"] > 1),
            }
        )
    return pd.DataFrame(rows)


def decoder_ablation(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare decoder variants: selection criterion plus per-series means."""
    keys = ["colour_depth", "cell_px", "series", "severity"]
    best = summary.groupby(keys)["goodput_mbps"].transform("max")
    n_cases = summary.groupby(keys).ngroups
    selection = pd.DataFrame(
        {
            "total loss (Mbit/s)": (best - summary["goodput_mbps"]).groupby(summary["decoder"]).sum(),
            "best in share of cases": summary[summary["goodput_mbps"] >= best - 1e-9].groupby("decoder").size() / n_cases,
        }
    ).sort_values("total loss (Mbit/s)").reset_index()
    return {
        "selection": selection,
        "goodput": summary.pivot_table(index="series", columns="decoder", values="goodput_mbps", aggfunc="mean"),
        "colour_ser": summary[summary["colour_depth"] > 1].pivot_table(index="series", columns="decoder", values="colour_ser", aggfunc="mean"),
        "shape_ser": summary.pivot_table(index="series", columns="decoder", values="shape_ser", aggfunc="mean"),
    }


def decomposition_cell_px(cfg: PilotConfig) -> int:
    """Cell size shown in the error-decomposition figure: the median configured size."""
    sizes = sorted(cfg.cell_px)
    return sizes[len(sizes) // 2]


def _cfg_name(row: pd.Series) -> str:
    return f"depth {int(row['colour_depth'])}, {int(row['cell_px'])} px"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

_INTERP = re.compile(r"<!-- interpretation -->.*?<!-- /interpretation -->", re.S)
_INTERP_PLACEHOLDER = "<!-- interpretation -->\n_(interpretation not yet written)_\n<!-- /interpretation -->"


def _md_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float | np.floating):
                cells.append(format(v, floatfmt.get(c, ".3g")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _series_line(s: Series) -> str:
    line = f"  * `{s.name}`: {SEVERITY_UNITS[s.degradation]}; severities {', '.join(f'{v:g}' for v in s.severities)}"
    return line + (f"; options `{dict(s.options)}`" if s.options else "")


def _grid(summary: pd.DataFrame, series: str, severity: float, decoder: str, value: str, fmt: str) -> str:
    s = summary[(summary["series"] == series) & (summary["severity"] == severity) & (summary["decoder"] == decoder)]
    table = s.pivot(index="colour_depth", columns="cell_px", values=value)
    best = s.loc[s["goodput_mbps"].idxmax()] if value == "goodput_mbps" else None
    lines = ["| colour depth \\ cell px | " + " | ".join(str(c) for c in table.columns) + " |",
             "|---|" + "|".join("---:" for _ in table.columns) + "|"]
    for depth, row in table.iterrows():
        cells = []
        for c, v in row.items():
            text = format(v, fmt)
            if best is not None and depth == best["colour_depth"] and c == best["cell_px"]:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {depth} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ser_table(summary: pd.DataFrame, series: str, decoder: str) -> str:
    """Rows = configuration, columns = severity, cell = 'shape% / colour%'."""
    s = summary[(summary["series"].isin([series, CLEAN])) & (summary["decoder"] == decoder)]
    sevs = sorted(s[s["series"] == series]["severity"].unique())
    if series.startswith("chroma"):
        s = s[s["series"] == series]
    else:
        sevs = [0.0, *sevs]
    header = "| config | " + " | ".join(f"{v:g}" for v in sevs) + " |"
    lines = [header, "|---|" + "|".join("---:" for _ in sevs) + "|"]
    for (depth, px), g in s.groupby(["colour_depth", "cell_px"]):
        by = {row["severity"]: row for _, row in g.iterrows()}
        cells = []
        for v in sevs:
            r = by.get(v)
            cells.append("–" if r is None else f"{100 * r['shape_ser']:.2f} / {100 * r['colour_ser']:.2f}")
        lines.append(f"| d{depth} {px}px | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _goodput_table(summary: pd.DataFrame, series: str, decoder: str) -> str:
    s = summary[(summary["series"].isin([series, CLEAN])) & (summary["decoder"] == decoder)]
    sevs = sorted(s[s["series"] == series]["severity"].unique())
    if series.startswith("chroma"):
        s = s[s["series"] == series]
    else:
        sevs = [0.0, *sevs]
    lines = ["| config | " + " | ".join(f"{v:g}" for v in sevs) + " |", "|---|" + "|".join("---:" for _ in sevs) + "|"]
    for (depth, px), g in s.groupby(["colour_depth", "cell_px"]):
        by = {row["severity"]: row for _, row in g.iterrows()}
        lines.append(f"| d{depth} {px}px | " + " | ".join(f"{by[v]['goodput_mbps']:.2f}" if v in by else "–" for v in sevs) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame, cfg: PilotConfig, path: Path, fig_dir: Path, elapsed_s: float, config_path: str
) -> None:
    head = cfg.headline_decoder.label
    win = winners(summary, head)
    rel = os.path.relpath(fig_dir, path.parent)
    existing = path.read_text() if path.exists() else ""
    match = _INTERP.search(existing)
    interpretation = match.group(0) if match else _INTERP_PLACEHOLDER

    def fig(name: str, caption: str) -> str:
        return f"![{caption}]({rel}/{name}.svg)\n\n*{caption}* ([PDF]({rel}/{name}.pdf))"

    chroma2 = win[(win["series"].str.startswith("chroma")) & (win["severity"] == 2.0)]
    blur = win[win["series"] == "blur"]
    colour_wins = win[win["colour_wins"]]
    clean = win[win["series"] == CLEAN].iloc[0]

    def fmt_w(df: pd.DataFrame) -> str:
        cols = ["series", "severity", "winner", "winner_mbps", "winner_rs", "best_mono", "mono_mbps", "best_colour", "colour_mbps", "colour_over_mono"]
        return _md_table(df[cols], {"winner_mbps": ".2f", "mono_mbps": ".2f", "colour_mbps": ".2f", "colour_over_mono": ".2f", "severity": "g"})

    ablation = decoder_ablation(summary)

    mono_wins = win[~win["colour_wins"]]
    per_series = []
    for series_name, g in win.groupby("series", sort=False):
        per_series.append(
            {
                "series": series_name,
                "colour wins at": ", ".join(f"{v:g}" for v in g[g["colour_wins"]]["severity"]) or "–",
                "monochrome wins at": ", ".join(f"{v:g}" for v in g[~g["colour_wins"]]["severity"]) or "–",
                "winning depths": ", ".join(sorted({w.split(",")[0].replace("depth ", "") for w in g["winner"]}, key=int)),
            }
        )
    colour_where = (
        f"A colour depth above 1 maximises goodput in **{len(colour_wins)} of {len(win)}** conditions"
        + (
            "; monochrome wins only at "
            + ", ".join(f"`{r.series}` severity {r.severity:g}" for r in mono_wins.itertuples())
            + "."
            if len(mono_wins)
            else "; monochrome never wins."
        )
        + "\n\n"
        + _md_table(pd.DataFrame(per_series))
    )
    sensitivity = pd.DataFrame(
        [
            {
                "decoder": d.label,
                "colour wins": f"{int(winners(summary, d.label)['colour_wins'].sum())} / {len(win)}",
                "monochrome wins at": ", ".join(
                    f"{r.series}@{r.severity:g}" for r in winners(summary, d.label).itertuples() if not r.colour_wins
                ) or "–",
            }
            for d in cfg.decoders
        ]
    )
    colour_where += "\n\nThe verdict under every decoder variant:\n\n" + _md_table(sensitivity)

    sections = [
        "**These are simulated results: they predict what the capture study should find; they do not measure it.** "
        "No camera, display or optics was involved; every number below comes from rendering, synthetically degrading "
        "and decoding frames in software.",
        "# Pilot study: simulated goodput vs colour depth and cell size",
        f"Generated by `python -m prism_share.sim.pilot --config {config_path}` "
        f"({len(cfg.configs())} configurations, {len(_conditions(cfg))} conditions, {cfg.frames} frames per condition, "
        f"{len(cfg.decoders)} decoder variants; {elapsed_s / 60:.1f} min on {os.cpu_count()} cores). "
        f"Full per-condition results: [`pilot_summary.csv`]({rel}/pilot_summary.csv).",
        "## Method",
        "\n".join(
            [
                "* **Configurations:** colour depth ∈ {" + ", ".join(map(str, cfg.colour_depths)) + "} × cell size ∈ {"
                + ", ".join(map(str, cfg.cell_px)) + "} px, gap 1 px, 16 glyphs, 1024 px frames, seed "
                + f"{cfg.seed}.",
                f"* **Frames:** {cfg.frames} real encoded frames per configuration (pseudo-random payload); each is "
                "degraded, quantised to 8 bits and decoded once per condition and decoder.",
                "* **Degradations, one at a time** (`prism_share/sim/degrade.py`):",
                *[_series_line(s) for s in cfg.series],
                "* **Metrics:** shape SER and colour SER per cell, separately (`analysis/metrics.py`).",
                f"* **Goodput** = payload bytes per frame × frame yield × {ASSUMED_FPS:g} fps (assumed, never timed). "
                "A frame counts toward yield only if every RS codeword satisfies 2·errors ≤ n − k (`analysis/ecc_sim.py`).",
                f"* **ECC:** reported two ways: the default RS(155,125), and the goodput-maximising RS(n, k) over "
                f"n ∈ {{{', '.join(map(str, cfg.ecc_lengths))}}} and every k (ECC is simulated, so every configuration "
                "gets its best code). Unless stated, goodput means the latter. It is an in-sample optimum over "
                f"{cfg.frames} frames and therefore slightly optimistic near the yield cliff.",
                f"* **Headline decoder:** `{head}` (see the decoder ablation).",
            ]
        ),
        "## Answers",
        "### Which configuration maximises goodput under chroma subsampling alone?",
        "At a chroma pitch of 2 screen px (4:2:0 with the camera sampling the screen 1:1), for each upsampling path and matrix:\n\n"
        + fmt_w(chroma2),
        "Across the whole chroma ladder (pitch 1 = YUV quantisation only; pitch < 2 models a camera that magnifies the screen):\n\n"
        + fmt_w(win[win["series"].isin(["chroma_nearest", "chroma_bilinear"])]),
        "### Which configuration maximises goodput under blur alone?",
        fmt_w(blur),
        "### Does any colour depth above 1 ever win, and where?",
        colour_where,
        f"For reference, undegraded frames: the winner is **{clean['winner']}** at {clean['winner_mbps']:.2f} Mbit/s "
        f"(best monochrome {clean['best_mono']} at {clean['mono_mbps']:.2f}).",
        "### All conditions",
        fmt_w(win),
        "## Interpretation",
        interpretation,
        "## Figures",
        fig("fig_best_goodput", "Best goodput over cell sizes, per colour depth, against severity. Monochrome in orange."),
        fig("fig_error_decomposition", f"Shape SER (top) and colour SER (bottom) against severity, at {decomposition_cell_px(cfg)} px cells."),
        fig("fig_goodput_surface", "Goodput (Mbit/s) over colour depth × cell size for three conditions; outlined cell = maximum."),
        "## Goodput grids (Mbit/s, best ECC; bold = maximum)",
        "### Clean\n\n" + _grid(summary, CLEAN, 0.0, head, "goodput_mbps", ".2f"),
        "### Chroma 4:2:0 (pitch 2), nearest upsampling\n\n" + _grid(summary, "chroma_nearest", 2.0, head, "goodput_mbps", ".2f"),
        "### Chroma 4:2:0 (pitch 2), bilinear upsampling\n\n" + _grid(summary, "chroma_bilinear", 2.0, head, "goodput_mbps", ".2f"),
        "### Blur σ = 1.0 px\n\n" + _grid(summary, "blur", 1.0, head, "goodput_mbps", ".2f"),
        "## Decoder ablation",
        "`max`/`luma` = the channel shape is read from (and that ranks a cell's pixels as ink); `mean`/`saturated` = "
        "colour from all ink pixels or from their most saturated quarter. The headline decoder is the one with the "
        "smallest total goodput loss against the best variant in each (configuration, condition).\n\n"
        + _md_table(ablation["selection"], {"total loss (Mbit/s)": ".1f", "best in share of cases": ".3f"})
        + "\n\nMean goodput (Mbit/s) over all configurations and severities, per series:\n\n"
        + _md_table(ablation["goodput"].reset_index(), {c: ".3f" for c in ablation["goodput"].columns})
        + "\n\nMean colour SER (colour depth > 1), per series:\n\n"
        + _md_table(ablation["colour_ser"].reset_index(), {c: ".4f" for c in ablation["colour_ser"].columns})
        + "\n\nMean shape SER, per series:\n\n"
        + _md_table(ablation["shape_ser"].reset_index(), {c: ".4f" for c in ablation["shape_ser"].columns}),
        "## Appendix A: shape SER / colour SER (%) per configuration, degradation and severity",
        f"Decoder `{head}`. Each cell is `shape % / colour %`; severity 0 is the undegraded frame.",
        *[f"### {s.name}\n\n{_ser_table(summary, s.name, head)}" for s in cfg.series],
        "## Appendix B: goodput (Mbit/s, best ECC) per configuration, degradation and severity",
        *[f"### {s.name}\n\n{_goodput_table(summary, s.name, head)}" for s in cfg.series],
    ]
    path.write_text("\n\n".join(sections) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="experiments/pilot.yaml")
    parser.add_argument("--out", default="docs/pilot", help="directory for summary CSV and figures")
    parser.add_argument("--report", default="docs/pilot.md")
    parser.add_argument("--data", default="data/pilot", help="directory for the frame-level Parquet (gitignored)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--reuse", action="store_true", help="skip simulation; rebuild report from existing Parquet")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    frames_path = data / "pilot_frames.parquet"

    elapsed_path = data / "elapsed_s.txt"
    if args.reuse and frames_path.exists():
        frames = pd.read_parquet(frames_path)
    else:
        print(f"simulating {len(cfg.configs())} configurations on {args.jobs} workers ...")
        started = time.monotonic()
        frames = run(cfg, args.jobs)
        frames.to_parquet(frames_path, index=False)
        elapsed_path.write_text(f"{time.monotonic() - started:.1f}\n")
    elapsed = float(elapsed_path.read_text()) if elapsed_path.exists() else 0.0

    summary = summarise(frames, cfg)
    summary.to_csv(out / "pilot_summary.csv", index=False, float_format="%.6g")

    from prism_share.analysis import plots

    plots.pilot_figures(summary, cfg.headline_decoder.label, out, decomposition_cell_px(cfg))
    write_report(summary, cfg, Path(args.report), out, elapsed, args.config)
    print(f"wrote {args.report}, {out}/pilot_summary.csv and figures")


if __name__ == "__main__":
    main()
