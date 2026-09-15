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
import dataclasses
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
from prism_share.codec.decoder import HEADLINE_DECODER, DecoderOptions, read_symbols
from prism_share.codec.encoder import EncodedFrame, band_index, render_frame
from prism_share.codec.framing import FrameHeader, encode_frame_symbols, frame_capacity
from prism_share.codec.params import (
    ASSUMED_FPS,
    BITS_PER_BYTE,
    HEADLINE_DECODER_REGISTERED,
    NEAR_TIE_MARGIN_PCT,
    CodecParams,
)
from prism_share.codec.layout import band_cell_cost
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
    #: If set, simulate only these (series, severity) conditions (``--add-missing``).
    only_conditions: frozenset[tuple[str, float]] | None = None

    def configs(self) -> list[CodecParams]:
        return [
            CodecParams(colour_depth=d, cell_px=c, seed=self.seed)
            for d, c in itertools.product(self.colour_depths, self.cell_px)
        ]


def _decoders(doc: dict[str, Any], path: Path) -> tuple[DecoderOptions, ...]:
    """Decoder variants to run. The headline is pre-registered (params.HEADLINE_*) and must be among them."""
    if "headline_decoder" in doc:
        raise ValueError(f"{path}: 'headline_decoder' is pre-registered in params.py (README 8.1) and may not be "
                         "set per experiment; remove the key")
    decoders = tuple(DecoderOptions(**d) for d in doc["decoders"])
    if HEADLINE_DECODER not in decoders:
        raise ValueError(f"{path}: decoders must include the pre-registered headline {HEADLINE_DECODER.label}")
    return (HEADLINE_DECODER,) + tuple(d for d in decoders if d != HEADLINE_DECODER)


def load_config(path: str | Path) -> PilotConfig:
    doc = yaml.safe_load(Path(path).read_text())
    return PilotConfig(
        seed=int(doc["seed"]),
        frames=int(doc["frames_per_condition"]),
        colour_depths=tuple(int(d) for d in doc["colour_depths"]),
        cell_px=tuple(int(c) for c in doc["cell_px"]),
        ecc_lengths=tuple(int(n) for n in doc["ecc_lengths"]),
        decoders=_decoders(doc, Path(path)),
        headline_decoder=HEADLINE_DECODER,
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
    if cfg.only_conditions is not None:
        out = [(name, step) for name, step in out if (name, step.severity if step else 0.0) in cfg.only_conditions]
    return out


def missing_conditions(frames: pd.DataFrame, cfg: PilotConfig) -> frozenset[tuple[str, float]]:
    """Conditions in ``cfg`` with no rows in ``frames``; refuses a table that does not match ``cfg`` otherwise.

    Adding conditions to an existing table is only sound because nothing a
    condition's frames depend on (payload, degradation seed = frame index,
    decoders) depends on which other conditions are simulated. Anything else
    that differs (frames, configurations, decoders, a condition the config no
    longer has, or a partially simulated condition) is an error, not a merge.
    """
    wanted = {(name, step.severity if step else 0.0) for name, step in _conditions(cfg)}
    have = {(str(a), float(b)) for a, b in frames[["series", "severity"]].drop_duplicates().itertuples(index=False)}
    if stale := have - wanted:
        raise ValueError(f"frame table has conditions the config does not: {sorted(stale)}")
    if set(frames["frame"].unique()) != set(range(cfg.frames)):
        raise ValueError(f"frame table does not have frames 0..{cfg.frames - 1}")
    if set(frames["decoder"].unique()) != {d.label for d in cfg.decoders}:
        raise ValueError("frame table was simulated with different decoders")
    configs = {(c.colour_depth, c.cell_px) for c in cfg.configs()}
    if {(int(a), int(b)) for a, b in frames[["colour_depth", "cell_px"]].drop_duplicates().itertuples(index=False)} != configs:
        raise ValueError("frame table was simulated with different configurations")
    expected_rows = cfg.frames * len(configs) * len(cfg.decoders)
    counts = frames.groupby(["series", "severity"]).size()
    if partial := [key for key, n in counts.items() if n != expected_rows]:
        raise ValueError(f"partially simulated conditions: {partial}")
    return frozenset(wanted - have)


def pilot_frame(params: CodecParams, frames: int, index: int) -> EncodedFrame:
    """Frame ``index`` of a ``frames``-frame pilot payload, built on its own.

    Equivalent to frame ``index`` of ``encode_frames`` over a payload of
    ``frames`` source blocks (a systematic frame carries its own block), but
    only one frame is ever in memory - at 200 frames per condition, building
    them all at once would hold ~600 MB per worker.
    """
    cap = frame_capacity(params)
    block = keystream(params.seed, "pilot-payload", cap.block_bytes, index=index)
    header = FrameHeader(index, frames, cap.block_bytes * frames, params.fingerprint())
    glyphs, colours = encode_frame_symbols(header, block, params)
    frame_index = band_index(index)
    return EncodedFrame(header, frame_index, glyphs, colours, render_frame(glyphs, colours, params, frame_index))


def simulate_config(params: CodecParams, cfg: PilotConfig) -> list[dict[str, Any]]:
    """All frames x conditions x decoders for one configuration (runs in a worker)."""
    cap = frame_capacity(params)
    conditions = _conditions(cfg)
    rows: list[dict[str, Any]] = []
    for index in range(cfg.frames):
        frame = pilot_frame(params, cfg.frames, index)
        for series_name, step in conditions:
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
        worst = {n: group[f"max_cw_errors_n{n}"].to_numpy() for n in cfg.ecc_lengths}
        evaluation = ~ecc_sim.selection_mask(group["frame"])
        # Every yield and goodput below is scored on the evaluation half only.
        fixed = ecc_sim.evaluate(group.loc[evaluation, f"max_cw_errors_n{params.ecc_total}"], params, params.ecc_total,
                                 params.ecc_data)
        record.update(
            fixed_rs=f"RS({fixed.n},{fixed.k})",
            fixed_yield=fixed.frame_yield,
            fixed_goodput_mbps=fixed.goodput_bytes_per_s * BITS_PER_BYTE / MBIT,
            fixed_goodput_band_credited_mbps=goodput_mbit_per_s(
                ecc_sim.payload_bytes_per_frame(params, fixed.n, fixed.k, band_credited=True), fixed.frame_yield),
        )
        oos = ecc_sim.out_of_sample(worst, group["frame"], params)
        best = oos.evaluated
        # Diagnostics only, never reported as goodput: the in-sample optimum on all frames, and on the
        # evaluation half itself (same frames as the reported number; the gap to it is pure selection optimism).
        in_sample = ecc_sim.best_code(worst, params)
        oracle = ecc_sim.best_code({n: w[evaluation] for n, w in worst.items()}, params)
        record.update(
            rs_selection="out_of_sample",
            n_selection_frames=oos.n_selection,
            n_evaluation_frames=oos.n_evaluation,
            best_rs=f"RS({best.n},{best.k})",
            best_code_rate=best.k / best.n,
            best_yield=best.frame_yield,
            best_payload_bytes=best.payload_bytes,
            goodput_mbps=best.goodput_bytes_per_s * BITS_PER_BYTE / MBIT,
            goodput_band_credited_mbps=goodput_mbit_per_s(
                ecc_sim.payload_bytes_per_frame(params, best.n, best.k, band_credited=True), best.frame_yield),
            in_sample_rs=f"RS({in_sample.n},{in_sample.k})",
            in_sample_goodput_mbps=in_sample.goodput_bytes_per_s * BITS_PER_BYTE / MBIT,
            evaluation_oracle_goodput_mbps=oracle.goodput_bytes_per_s * BITS_PER_BYTE / MBIT,
            band_cell_cost_pct=100 * band_cell_cost(params) / (frame_capacity(params).n_cells + band_cell_cost(params)),
            ceiling_mbps=goodput_mbit_per_s(frame_capacity(params).capacity_bytes, 1.0),
        )
        out.append(record)
    return pd.DataFrame(out)


#: The two goodput columns every winner is reported under.
GOODPUT_COLUMNS: dict[str, str] = {"measured": "goodput_mbps", "band credited": "goodput_band_credited_mbps"}


def margin_pct(better: float, worse: float) -> float:
    """How much ``better`` exceeds ``worse``, in percent of ``worse`` (inf if worse is 0, 0 if both are)."""
    if worse <= 0:
        return 0.0 if better <= 0 else float("inf")
    return 100.0 * (better - worse) / worse


def winners(summary: pd.DataFrame, decoder: str, column: str = "goodput_mbps") -> pd.DataFrame:
    """Per (series, severity): winner, runner-up and best of the other class, with margins.

    Two margins, both flagged below NEAR_TIE_MARGIN_PCT:
    * ``margin_pct``: winner over the runner-up (any configuration);
    * ``class_margin_pct``: winner over the best configuration of the *other*
      class (monochrome vs colour). This is the one that decides whether a
      condition counts towards "colour wins", so it is the one that says
      whether that count is a result or a coin flip.
    Ties in goodput go to fewer colours, then larger cells (the simpler code).
    """
    s = summary[summary["decoder"] == decoder]
    rows = []
    for (series, severity), g in s.groupby(["series", "severity"], sort=False):
        g = g.sort_values([column, "colour_depth", "cell_px"], ascending=[False, True, False])
        top, runner = g.iloc[0], g.iloc[1]
        colour_wins = bool(top["colour_depth"] > 1)
        other = g[(g["colour_depth"] == 1) if colour_wins else (g["colour_depth"] > 1)].iloc[0]
        mono = g[g["colour_depth"] == 1].iloc[0]
        colour = g[g["colour_depth"] > 1].iloc[0]
        margin = margin_pct(top[column], runner[column])
        class_margin = margin_pct(top[column], other[column])
        rows.append(
            {
                "series": series,
                "severity": severity,
                "winner": _cfg_name(top),
                "winner_mbps": top[column],
                "winner_rs": top["best_rs"],
                "runner_up": _cfg_name(runner),
                "runner_up_mbps": runner[column],
                "margin_pct": margin,
                "other_class_best": _cfg_name(other),
                "other_class_mbps": other[column],
                "class_margin_pct": class_margin,
                "best_mono": _cfg_name(mono),
                "mono_mbps": mono[column],
                "best_colour": _cfg_name(colour),
                "colour_mbps": colour[column],
                "colour_wins": colour_wins,
                "near_tie": margin < NEAR_TIE_MARGIN_PCT,
                "class_near_tie": class_margin < NEAR_TIE_MARGIN_PCT,
            }
        )
    return pd.DataFrame(rows)


def verdict_table(summary: pd.DataFrame, decoders: tuple[DecoderOptions, ...]) -> pd.DataFrame:
    """The colour-wins count under every decoder and both goodput columns, with how many are near-ties."""
    rows = []
    for d in decoders:
        for label, column in GOODPUT_COLUMNS.items():
            w = winners(summary, d.label, column)
            robust_colour = w["colour_wins"] & ~w["class_near_tie"]
            robust_mono = ~w["colour_wins"] & ~w["class_near_tie"]
            rows.append(
                {
                    "decoder": d.label + (" (pre-registered headline)" if d == HEADLINE_DECODER else " (sensitivity)"),
                    "goodput column": label,
                    "colour wins": f"{int(w['colour_wins'].sum())} / {len(w)}",
                    f"colour wins, class margin >= {NEAR_TIE_MARGIN_PCT:g} %": int(robust_colour.sum()),
                    "monochrome wins": int((~w["colour_wins"]).sum()),
                    f"monochrome wins, class margin >= {NEAR_TIE_MARGIN_PCT:g} %": int(robust_mono.sum()),
                    f"near-ties (class margin < {NEAR_TIE_MARGIN_PCT:g} %)": int(w["class_near_tie"].sum()),
                    "monochrome wins at": ", ".join(f"{r.series}@{r.severity:g}" for r in w.itertuples() if not r.colour_wins) or "–",
                }
            )
    return pd.DataFrame(rows)


#: Goodputs closer than this are the same number (they are sums of exact byte counts).
GOODPUT_TOLERANCE_MBPS = 1e-9


#: Diagnostic in-sample goodput: RS code chosen and scored on all frames. Never a reported goodput.
IN_SAMPLE_COLUMN = "in_sample_goodput_mbps"


def selection_bias(summary: pd.DataFrame, decoder: str) -> dict[str, Any]:
    """How reported (out-of-sample) goodput compares with the in-sample optimum, for one decoder.

    ``fall`` is in-sample minus out-of-sample, per (configuration, condition)
    cell. ``oracle_fall`` compares against the in-sample optimum on the
    evaluation half itself, so same frames and only the selection differs:
    that gap is pure selection optimism.
    """
    s = summary[summary["decoder"] == decoder]
    fall = s[IN_SAMPLE_COLUMN] - s["goodput_mbps"]
    positive = s[IN_SAMPLE_COLUMN] > 0
    changed = fall.abs() > GOODPUT_TOLERANCE_MBPS
    w_in, w_out = winners(summary, decoder, IN_SAMPLE_COLUMN), winners(summary, decoder)
    m = w_in.merge(w_out, on=["series", "severity"], suffixes=("_in", "_out"))
    return {
        "cells": len(s),
        "mean_in_sample_mbps": float(s[IN_SAMPLE_COLUMN].mean()),
        "mean_out_of_sample_mbps": float(s["goodput_mbps"].mean()),
        "mean_fall_mbps": float(fall.mean()),
        "mean_fall_pct": float(100 * (fall[positive] / s.loc[positive, IN_SAMPLE_COLUMN]).mean()),
        "mean_fall_pct_changed": float(100 * (fall[positive & changed] / s.loc[positive & changed, IN_SAMPLE_COLUMN]).mean())
        if (positive & changed).any() else 0.0,
        "max_fall_pct": float(100 * (fall[positive] / s.loc[positive, IN_SAMPLE_COLUMN]).max()),
        "mean_oracle_fall_mbps": float((s["evaluation_oracle_goodput_mbps"] - s["goodput_mbps"]).mean()),
        "fell": int((fall > GOODPUT_TOLERANCE_MBPS).sum()),
        "rose": int((fall < -GOODPUT_TOLERANCE_MBPS).sum()),
        "colour_wins_in_sample": int(w_in["colour_wins"].sum()),
        "colour_wins_out_of_sample": int(w_out["colour_wins"].sum()),
        "winner_changed": m[m["winner_in"] != m["winner_out"]],
        "class_changed": m[m["colour_wins_in"] != m["colour_wins_out"]],
    }


def chroma_ladder(summary: pd.DataFrame, decoder: str, series: str) -> tuple[pd.DataFrame, list[float]]:
    """Best colour vs best monochrome along one chroma-pitch series, and where they cross.

    ``colour_lead_pct`` = 100 * (best colour - best mono) / best mono. Crossings
    are linear interpolations of that lead between adjacent rungs where it
    changes sign. Every crossing is returned, because the lead need not be
    monotonic in pitch.
    """
    w = winners(summary, decoder)
    w = w[w["series"] == series].sort_values("severity")
    lead = 100 * (w["colour_mbps"] - w["mono_mbps"]) / w["mono_mbps"]
    table = w.assign(colour_lead_pct=lead.to_numpy())
    crossings = []
    pts = list(zip(table["severity"], table["colour_lead_pct"], strict=True))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 > 0) != (y1 > 0):
            crossings.append(float(x0 + (x1 - x0) * y0 / (y0 - y1)))
    return table, crossings


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
    best = s.loc[s[value].idxmax()] if value.startswith("goodput") else None
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


def _pct(v: float) -> str:
    return "∞" if v == float("inf") else f"{v:.1f} %"


def margin_table(win: pd.DataFrame) -> str:
    """The per-condition margin table: every winner with the evidence behind it, near-ties flagged."""
    lines = [
        "| condition | winner | Mbit/s | runner-up | Mbit/s | margin | best of other class | Mbit/s | class margin | flag |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for r in win.itertuples():
        flags = []
        if r.class_near_tie:
            flags.append("**class near-tie**")
        if r.near_tie:
            flags.append("near-tie")
        lines.append(
            f"| {r.series} @ {r.severity:g} | {r.winner} | {r.winner_mbps:.3f} | {r.runner_up} | {r.runner_up_mbps:.3f} | "
            f"{_pct(r.margin_pct)} | {r.other_class_best} | {r.other_class_mbps:.3f} | {_pct(r.class_margin_pct)} | "
            f"{', '.join(flags) or '–'} |"
        )
    return "\n".join(lines)


def credited_check_table(measured: pd.DataFrame, credited: pd.DataFrame) -> str:
    m = measured.merge(credited, on=["series", "severity"], suffixes=("_m", "_c"))
    lines = ["| condition | winner (measured) | winner (band credited) | same | class margin (credited) | flag |",
             "|---|---|---|---|---:|---|"]
    for r in m.itertuples():
        lines.append(f"| {r.series} @ {r.severity:g} | {r.winner_m} | {r.winner_c} | {'yes' if r.winner_m == r.winner_c else '**NO**'} | "
                     f"{_pct(r.class_margin_pct_c)} | {'**class near-tie**' if r.class_near_tie_c else '–'} |")
    return "\n".join(lines)


def _grid_both(summary: pd.DataFrame, series: str, severity: float, decoder: str) -> str:
    return "\n\n".join(
        f"*{label}:*\n\n" + _grid(summary, series, severity, decoder, column, ".2f")
        for label, column in GOODPUT_COLUMNS.items()
    )


#: Chroma rungs added on 2026-09-15; the earlier runs' counts are over the 34 conditions without them.
LADDER_ADDED_2026_09_15: frozenset[float] = frozenset({2.25, 2.5, 2.75, 3.25, 3.5, 3.75})
LADDER_ADDED_SERIES: frozenset[str] = frozenset({"chroma_nearest", "chroma_bilinear"})

REVISION_HISTORY = """Every count below is the number of **conditions in which a configuration with more than one colour maximises goodput**, under `luma/saturated`. The earlier runs had 34 conditions; the widened chroma ladder brings this report to {n_conditions}, so this report's count is also given over the original 34.

| Run | Frame format | Frames per condition | RS code chosen | Conditions | Colour wins | Colour wins over the original 34 | Monochrome wins at |
|---|---|---:|---|---:|---:|---:|---|
| 2026-09-11 | 1 (no index band) | 4 | in sample | 34 | 33 / 34 | 33 / 34 | chroma_bilinear @ 4 |
| 2026-09-12 | 2 (index band) | 4 | in sample | 34 | 32 / 34 | 32 / 34 | chroma_bilinear @ 4, chroma_nearest @ 4 |
| 2026-09-14 | 2 | 200 | in sample | 34 | 32 / 34 | 32 / 34 | chroma_bilinear @ 4, chroma_nearest @ 4 |
| {date} (this report) | 2 | {frames} | out of sample | {n_conditions} | {count} / {n_conditions} | {count34} / 34 | {mono_at} |

**Format 1 → format 2 (4 frames each).** One condition changed direction: chroma_nearest @ 4, where 2 colours at 4 px (4.571 Mbit/s) became beaten by monochrome at 4 px (4.393). It was **not** the index band's cell cost: with the band's cells credited back, monochrome still won that condition (4.504 vs 4.348). The 2-colour configuration's colour SER was essentially unchanged (3.36 % vs 3.30 %). What changed was the goodput-maximising code, RS(255,205) in format 1 and RS(255,195) in format 2, because that choice rests on the worst codeword across only 4 frames, and in format 1 colour led monochrome by just 1.4 %. Under the `mean` decoders the same mechanism flipped chroma_nearest @ 3 instead. This is why the pilot was re-run at more frames per condition and why every winner is now reported with its margin.

**4 → 200 frames (format 2, 2026-09-14).** The headline count did not move (32 / 34, monochrome at the same two conditions), and no condition changed class under `luma/saturated`. What changed is how settled it is. The one class near-tie at 4 frames, chroma_nearest @ 4 (monochrome ahead by 4.4 %), is now decided at 8.8 %: over 200 frames the 2-colour, 4 px code needs RS(255,191) instead of RS(255,195) and loses 2 % of frames, so its goodput fell from 4.207 to 4.038 Mbit/s. Its colour SER did not change (3.30 % vs 3.33 %). The larger sample removed optimism from the in-sample RS choice; it did not reveal a different error rate. Across all 850 (configuration, condition) cells of the headline decoder, goodput fell in 261, rose in 20 and was unchanged in the rest, which is that optimism draining out. One winner changed identity without changing class: perspective @ 0.3 went from 16 colours at 5 px to 16 colours at 6 px (a 0.6 % margin at 4 frames, 1.9 % now). Under the two `mean` sensitivity decoders, chroma_bilinear @ 3 moved from colour to monochrome (colour ahead by 3.3 % at 4 frames; monochrome ahead by about 2 % now), taking them from 31 to 30 of 34.

**In sample → out of sample, and a finer chroma ladder (2026-09-15).** The RS code was still being chosen on the frames it was scored on. At 200 frames that optimism was diluted, not removed. From this report on, the code is chosen on half the frames and scored on the other half (section "RS code chosen out of sample", which also gives the size of the removed bias). The chroma ladders gained rungs at 2.25, 2.5, 2.75, 3.25, 3.5 and 3.75 so the colour/monochrome crossing can be located, not only bracketed (section "Chroma-pitch ladder")."""


def selection_section(summary: pd.DataFrame, cfg: PilotConfig) -> str:
    head = cfg.headline_decoder.label
    b = selection_bias(summary, head)
    n_sel = int(summary["n_selection_frames"].iloc[0])
    n_eval = int(summary["n_evaluation_frames"].iloc[0])
    lines = [
        f"Every goodput in this report is **out of sample**. Each condition's {cfg.frames} frames are split by frame "
        f"position (even → selection, odd → evaluation; `params.RS_SELECTION_PERIOD`), so the split is the same for every "
        f"configuration in a condition and comparisons stay paired. The goodput-maximising RS(n, k) is chosen on the "
        f"{n_sel} selection frames. Yield and goodput are scored only on the {n_eval} evaluation frames, with that code. "
        "The fixed RS(155,125) columns are scored on the evaluation half too, so every yield in the tables uses the same frames.",
        "",
        "Choosing the code on the frames it is scored on (in sample) is shown here only to measure the bias that was removed:",
        "",
        "| decoder | mean goodput, in sample (all frames) | mean goodput, out of sample | mean fall | mean fall where it changed | largest fall | cells fell / rose / of | vs in-sample optimum on the evaluation half | colour wins in sample → out of sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in cfg.decoders:
        x = selection_bias(summary, d.label)
        role = " (headline)" if d == cfg.headline_decoder else ""
        lines.append(
            f"| {d.label}{role} | {x['mean_in_sample_mbps']:.4f} | {x['mean_out_of_sample_mbps']:.4f} | "
            f"{x['mean_fall_mbps']:.4f} Mbit/s ({x['mean_fall_pct']:.2f} %) | {x['mean_fall_pct_changed']:.2f} % | {x['max_fall_pct']:.1f} % | "
            f"{x['fell']} / {x['rose']} / {x['cells']} | {x['mean_oracle_fall_mbps']:.4f} Mbit/s | "
            f"{x['colour_wins_in_sample']} → {x['colour_wins_out_of_sample']} |"
        )
    lines.append("")
    changed = b["winner_changed"]
    if len(changed):
        lines += [f"Conditions whose winner differs between in-sample and out-of-sample scoring (`{head}`):", "",
                  "| condition | winner in sample | Mbit/s | winner out of sample | Mbit/s | class changed |", "|---|---|---:|---|---:|---|"]
        lines += [f"| {r.series} @ {r.severity:g} | {r.winner_in} | {r.winner_mbps_in:.3f} | {r.winner_out} | {r.winner_mbps_out:.3f} | "
                  f"{'**yes**' if r.colour_wins_in != r.colour_wins_out else 'no'} |" for r in changed.itertuples()]
    else:
        lines.append(f"No condition changes winner between in-sample and out-of-sample scoring under `{head}`.")
    lines.append("")
    lines.append(f"Conditions that change class (colour ↔ monochrome) under `{head}`: **{len(b['class_changed'])}**"
                 + (": " + ", ".join(f"{r.series} @ {r.severity:g}" for r in b["class_changed"].itertuples()) if len(b["class_changed"]) else "") + ".")
    return "\n".join(lines)


def ladder_section(summary: pd.DataFrame, cfg: PilotConfig) -> str:
    head = cfg.headline_decoder.label
    names = [sr.name for sr in cfg.series if sr.degradation == "chroma" and len(sr.severities) > 2]
    tables = {n: chroma_ladder(summary, head, n) for n in names}
    lines = [
        "Best colour configuration against best monochrome along the chroma-pitch ladder. *Colour lead* = (best colour − "
        "best monochrome) / best monochrome. Positive means colour wins. A crossing is a linear interpolation of the lead "
        "between adjacent rungs, so it is only as trustworthy as the lead is smooth between them.",
        "",
        "| pitch | " + " | ".join(f"{n}: winner | colour lead" for n in names) + " |",
        "|---:|" + "|".join("---|---:" for _ in names) + "|",
    ]
    pitches = sorted(set().union(*(set(t["severity"]) for t, _ in tables.values())))
    for x in pitches:
        cells = []
        for n in names:
            row = tables[n][0][tables[n][0]["severity"] == x]
            cells.append("– | –" if row.empty else f"{row['winner'].iloc[0]} | {row['colour_lead_pct'].iloc[0]:+.1f} %")
        lines.append(f"| {x:g} | " + " | ".join(cells) + " |")
    lines.append("")

    def describe(table: pd.DataFrame, crossings: list[float]) -> str:
        if not crossings:
            return "no crossing on this ladder"
        rungs = list(table["severity"])
        brackets = [f"between {max(r for r in rungs if r <= c):g} and {min(r for r in rungs if r >= c):g}" for c in crossings]
        return "; ".join(f"≈ {c:.2f} ({b})" for c, b in zip(crossings, brackets, strict=True))

    lines += [f"Where colour and monochrome cross (interpolated pitch, and the rungs that bracket it), `{head}`:", ""]
    lines += [f"* **{n}:** {describe(*tables[n])}." for n in names]
    lines += ["", "Sensitivity: the same crossing under every decoder.", "",
              "| decoder | " + " | ".join(names) + " |", "|---|" + "|".join("---" for _ in names) + "|"]
    for d in cfg.decoders:
        role = " (headline)" if d == cfg.headline_decoder else ""
        lines.append(f"| {d.label}{role} | " + " | ".join(describe(*chroma_ladder(summary, d.label, n)) for n in names) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame, cfg: PilotConfig, path: Path, fig_dir: Path, elapsed_s: float, config_path: str
) -> None:
    head = cfg.headline_decoder.label
    win = winners(summary, head)
    win_credited = winners(summary, head, "goodput_band_credited_mbps")
    rel = os.path.relpath(fig_dir, path.parent)
    existing = path.read_text() if path.exists() else ""
    match = _INTERP.search(existing)
    interpretation = match.group(0) if match else _INTERP_PLACEHOLDER

    def fig(name: str, caption: str) -> str:
        return f"![{caption}]({rel}/{name}.svg)\n\n*{caption}* ([PDF]({rel}/{name}.pdf))"

    n = len(win)
    colour_wins = int(win["colour_wins"].sum())
    colour_wins_credited = int(win_credited["colour_wins"].sum())
    robust_colour = int((win["colour_wins"] & ~win["class_near_tie"]).sum())
    robust_mono = int((~win["colour_wins"] & ~win["class_near_tie"]).sum())
    ties = win[win["class_near_tie"]]
    close = win[win["near_tie"]]
    is_added = win["series"].isin(LADDER_ADDED_SERIES) & win["severity"].isin(LADDER_ADDED_2026_09_15)
    same = int((win["winner"].to_numpy() == win_credited["winner"].to_numpy()).sum())
    mono_at = ", ".join(f"{r.series} @ {r.severity:g}" for r in win.itertuples() if not r.colour_wins) or "none"
    ablation = decoder_ablation(summary)
    band_cost = summary.groupby("cell_px")["band_cell_cost_pct"].first()

    sections = [
        "**These are simulated results: they predict what the capture study should find; they do not measure it.** "
        "No camera, display or optics was involved; every number below comes from rendering, synthetically degrading "
        "and decoding frames in software.",
        "# Pilot study: simulated goodput vs colour depth and cell size",
        f"Generated by `python -m prism_share.sim.pilot --config {config_path}` "
        f"({len(cfg.configs())} configurations, {len(_conditions(cfg))} conditions, **{cfg.frames} frames per condition**, "
        f"{len(cfg.decoders)} decoder variants; {elapsed_s / 60:.0f} min on {os.cpu_count()} cores). "
        f"Full per-condition results: [`pilot_summary.csv`]({rel}/pilot_summary.csv).",
        "## Headline",
        f"* **Decoder:** `{head}`, pre-registered on {HEADLINE_DECODER_REGISTERED} before any real capture, for reasons "
        "stated in README section 8.1 that do not depend on these results. The other three decoders are sensitivity "
        "analysis only.\n"
        f"* **Count:** a configuration with more than one colour maximises goodput in **{colour_wins} of {n}** conditions "
        f"as measured, and **{colour_wins_credited} of {n}** with the index band's cells credited back. The winning "
        f"configuration is identical in both columns in {same} of {n} conditions.\n"
        f"* **How much of that is decided:** a condition counts as decided only if the winner beats the best "
        f"configuration of the other class (monochrome vs colour) by at least {NEAR_TIE_MARGIN_PCT:g} %. "
        f"Decided colour wins: **{robust_colour}**. Decided monochrome wins: **{robust_mono}**. "
        f"Near-ties: **{len(ties)}**"
        + (": " + ", ".join(f"{r.series} @ {r.severity:g} ({_pct(r.class_margin_pct)})" for r in ties.itertuples()) if len(ties) else "")
        + ".\n"
        f"* **Winner vs runner-up (any configuration) under {NEAR_TIE_MARGIN_PCT:g} %:** **{len(close)}**"
        + (": " + ", ".join(f"{r.series} @ {r.severity:g} ({r.winner} vs {r.runner_up}, {_pct(r.margin_pct)})" for r in close.itertuples()) if len(close) else "")
        + ". These do not move the count unless the runner-up is of the other class, which the class margin above already covers; "
        "they mean the *identity* of the best configuration is not settled there.\n"
        f"* **Monochrome wins at:** {mono_at}.",
        "## RS code chosen out of sample",
        selection_section(summary, cfg),
        "## Chroma-pitch ladder",
        ladder_section(summary, cfg),
        "## Margin table (pre-registered decoder, goodput as measured)",
        f"*margin* = how much the winner beats the runner-up (any configuration); *class margin* = how much it beats "
        f"the best configuration of the other class, the margin that decides the count. Both in percent of the loser's "
        f"goodput. Flags mark margins under {NEAR_TIE_MARGIN_PCT:g} %. Goodput is out of sample: the RS code is chosen "
        f"on the selection half and scored on the evaluation half.\n\n" + margin_table(win),
        "## Band-credited check (pre-registered decoder)",
        "The index band is measurement apparatus, not codec, and its cell cost varies with cell size ("
        + ", ".join(f"{pct:.2f} % at {px} px" for px, pct in band_cost.items())
        + "). This table re-ranks every condition with the band's cells credited back (same RS code, same "
        "yield, payload of the band-free grid).\n\n" + credited_check_table(win, win_credited),
        "## Sensitivity: every decoder, both goodput columns",
        _md_table(verdict_table(summary, cfg.decoders)),
        "## Interpretation",
        interpretation,
        "## Revision history",
        REVISION_HISTORY.format(date=pd.Timestamp.now().strftime("%Y-%m-%d"), frames=cfg.frames, count=colour_wins, mono_at=mono_at,
                                n_conditions=n, count34=int(win.loc[~is_added, "colour_wins"].sum())),
        "## Method",
        "\n".join(
            [
                "* **Configurations:** colour depth ∈ {" + ", ".join(map(str, cfg.colour_depths)) + "} × cell size ∈ {"
                + ", ".join(map(str, cfg.cell_px)) + "} px, gap 1 px, 16 glyphs, 1024 px frames (format 2, with index "
                f"band), seed {cfg.seed}.",
                f"* **Frames:** {cfg.frames} distinct encoded frames per configuration (pseudo-random payload, one "
                "source block each); each is degraded, quantised to 8 bits and decoded once per condition and decoder.",
                "* **Degradations, one at a time** (`prism_share/sim/degrade.py`):",
                *[_series_line(sr) for sr in cfg.series],
                "* **Metrics:** shape SER and colour SER per cell, separately (`analysis/metrics.py`).",
                f"* **Goodput** = payload bytes per frame × frame yield × {ASSUMED_FPS:g} fps (assumed, never timed). "
                "A frame counts toward yield only if every RS codeword satisfies 2·errors ≤ n − k (`analysis/ecc_sim.py`).",
                "* **Two goodput columns:** as measured (frame with the index band), and with the band's cells credited "
                "back (same code and yield, payload of the grid without the band).",
                f"* **ECC:** the goodput-maximising RS(n, k) over n ∈ {{{', '.join(map(str, cfg.ecc_lengths))}}} and every k, "
                "chosen on the selection half (even frame positions) and scored on the evaluation half (odd). Fixed "
                "RS(155,125) is also scored on the evaluation half. SERs are not selected on anything and use all frames.",
                f"* **Near-tie threshold:** {NEAR_TIE_MARGIN_PCT:g} % (`params.NEAR_TIE_MARGIN_PCT`).",
            ]
        ),
        "## Figures",
        fig("fig_best_goodput", "Best goodput over cell sizes, per colour depth, against severity (as measured). Monochrome in orange."),
        fig("fig_error_decomposition", f"Shape SER (top) and colour SER (bottom) against severity, at {decomposition_cell_px(cfg)} px cells."),
        fig("fig_goodput_surface", "Goodput (Mbit/s, as measured) over colour depth × cell size for three conditions; outlined cell = maximum."),
        "## Goodput grids (Mbit/s, best ECC; bold = maximum)",
        "### Clean\n\n" + _grid_both(summary, CLEAN, 0.0, head),
        "### Chroma 4:2:0 (pitch 2), nearest upsampling\n\n" + _grid_both(summary, "chroma_nearest", 2.0, head),
        "### Chroma 4:2:0 (pitch 2), bilinear upsampling\n\n" + _grid_both(summary, "chroma_bilinear", 2.0, head),
        "### Blur σ = 1.0 px\n\n" + _grid_both(summary, "blur", 1.0, head),
        "## Decoder sensitivity data",
        "Descriptive only: the headline decoder is pre-registered, not selected from these numbers. `max`/`luma` = the "
        "channel shape is read from (and that ranks a cell's pixels as ink); `mean`/`saturated` = colour from all ink "
        "pixels or from their most saturated quarter. *Total loss* = goodput lost against the best variant in each "
        "(configuration, condition).\n\n"
        + _md_table(ablation["selection"], {"total loss (Mbit/s)": ".1f", "best in share of cases": ".3f"})
        + "\n\nMean goodput (Mbit/s) over all configurations and severities, per series:\n\n"
        + _md_table(ablation["goodput"].reset_index(), {c: ".3f" for c in ablation["goodput"].columns})
        + "\n\nMean colour SER (colour depth > 1), per series:\n\n"
        + _md_table(ablation["colour_ser"].reset_index(), {c: ".4f" for c in ablation["colour_ser"].columns})
        + "\n\nMean shape SER, per series:\n\n"
        + _md_table(ablation["shape_ser"].reset_index(), {c: ".4f" for c in ablation["shape_ser"].columns}),
        "## Appendix A: shape SER / colour SER (%) per configuration, degradation and severity",
        f"Decoder `{head}`. Each cell is `shape % / colour %`; severity 0 is the undegraded frame.",
        *[f"### {sr.name}\n\n{_ser_table(summary, sr.name, head)}" for sr in cfg.series],
        "## Appendix B: goodput (Mbit/s, best ECC, as measured) per configuration, degradation and severity",
        *[f"### {sr.name}\n\n{_goodput_table(summary, sr.name, head)}" for sr in cfg.series],
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
    parser.add_argument("--add-missing", action="store_true",
                        help="simulate only conditions absent from the existing Parquet and append them")
    parser.add_argument("--simulate-only", action="store_true", help="write the frame-level Parquet and stop")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    frames_path = data / "pilot_frames.parquet"

    elapsed_path = data / "elapsed_s.txt"
    if args.reuse and frames_path.exists():
        frames = pd.read_parquet(frames_path)
    elif args.add_missing and frames_path.exists():
        existing = pd.read_parquet(frames_path)
        missing = missing_conditions(existing, cfg)
        print(f"adding {len(missing)} missing conditions {sorted(missing)} on {args.jobs} workers ...")
        started = time.monotonic()
        added = run(dataclasses.replace(cfg, only_conditions=missing), args.jobs) if missing else existing.iloc[:0]
        frames = pd.concat([existing, added], ignore_index=True)
        frames.to_parquet(frames_path, index=False)
        previous = float(elapsed_path.read_text()) if elapsed_path.exists() else 0.0
        elapsed_path.write_text(f"{previous + time.monotonic() - started:.1f}\n")
        if args.simulate_only:
            print(f"wrote {frames_path} ({len(frames)} rows); report not generated (--simulate-only)")
            return
    else:
        print(f"simulating {len(cfg.configs())} configurations on {args.jobs} workers ...")
        started = time.monotonic()
        frames = run(cfg, args.jobs)
        frames.to_parquet(frames_path, index=False)
        elapsed_path.write_text(f"{time.monotonic() - started:.1f}\n")
        if args.simulate_only:
            print(f"wrote {frames_path} ({len(frames)} rows); report not generated (--simulate-only)")
            return
    elapsed = float(elapsed_path.read_text()) if elapsed_path.exists() else 0.0

    summary = summarise(frames, cfg)
    summary.to_csv(out / "pilot_summary.csv", index=False, float_format="%.6g")

    from prism_share.analysis import plots

    plots.pilot_figures(summary, cfg.headline_decoder.label, out, decomposition_cell_px(cfg))
    write_report(summary, cfg, Path(args.report), out, elapsed, args.config)
    print(f"wrote {args.report}, {out}/pilot_summary.csv and figures")


if __name__ == "__main__":
    main()
