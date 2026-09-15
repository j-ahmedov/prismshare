"""Captured-data sweep (build step 6): ingested runs -> tidy tables -> Parquet (+ figures).

    python -m prism_share.analysis.sweep data/<run_id> [data/<run_id> ...] --out data/sweep [--figures docs/figures]

Per run (``process_run``), in two passes over the untainted frames:

1. **Detect** each capture once, on its Y plane (the JPEG image on that path),
   then read its **index band**: the frame says which frame it is. Index 0 is
   the reference frame, so reference captures are recognised structurally, not
   by their appearance; they are excluded from every yield and, if
   ``flat_field`` is on, the first one becomes the run's flat field (valid only
   because exposure is locked for the whole run).
2. **Decode** each code capture from every pixel source (``PIXEL_SOURCES``;
   'y' only for monochrome, 'jpeg' on the JPEG path) with every decoder
   variant, and compare with the ground truth of the frame the band names:
   shape SER, colour SER, and the stream byte-error pattern, reduced to the
   maximum per-codeword error count for each codeword length.

The band is read after rectification, from blocks two orders of magnitude
larger in area than a data cell, so identification does **not** depend on how
well the data decoded. That matters: identifying a capture by how well it
matches candidate frames would exclude exactly the badly degraded captures,
making exclusion correlate with the outcome being measured and compressing the
difference between configurations by a configuration-dependent amount.

Every code frame then has exactly one of three outcomes under a given RS(n, k):
**detection failed**, **detected but not recoverable** (including a capture
whose band names a frame other than the run's one static frame), or
**recovered**. Frames below
``min_source_px_per_cell`` are excluded and counted, never silently mixed in.

**Two goodput columns.** The index band is measurement apparatus, not codec:
a deployed system carries its frame index in the fountain header. Its cell
cost varies with cell_px (2.83 % at 4 px, 3.57 % at 10 px), so it would enter
the between-configuration comparison as a term that is not physics. Every
goodput is therefore reported twice: ``*_goodput_mbps`` as measured, and
``*_goodput_band_credited_mbps`` with the band's cells credited back (same RS
code, same measured yield, payload recomputed for the grid without the band).
``band_cells_lost`` and ``band_cell_cost_pct`` state the cost per configuration,
and ``winners_both_columns`` checks that the winning configuration is the same
in both columns.

**RS codes are chosen out of sample.** A run's code captures are split by
their capture ``index`` (``params.RS_SELECTION_PERIOD``: even indices select,
odd indices evaluate). The goodput-maximising RS(n, k) is chosen on the
selection half, and every ``best_*`` and ``fixed_*`` outcome, yield and goodput
is counted on the evaluation half only. The split depends on capture position
alone, which is fixed before any analysis, so it is the same rule for every
configuration. A run with no frames in one half gets NaN goodput, never an
in-sample number. ``SweepConfig(rs_selection="in_sample")`` (CLI
``--in-sample``) chooses and scores on all frames. It exists only to measure the
bias, it warns, and every row and figure it produces says ``in_sample``.

Outputs: ``frames.parquet`` (one row per capture x pixel source x decoder),
``summary.parquet`` / ``summary.csv`` (one row per run x pixel source x
decoder: the run fixes configuration, condition and device) and
``winners.csv`` (the winner under each goodput column), each row carrying its
provenance: run id, device, capture resolution, YUV matrix/range and whether
the range was read or assumed.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from prism_share.analysis import ecc_sim
from prism_share.analysis.detect import (
    Detection,
    FlatField,
    FrontEnd,
    detect,
    locate_fiducials,
    rectify_with,
    write_overlay,
)
from prism_share.analysis.metrics import (
    byte_errors,
    frame_outcome,
    goodput_mbit_per_s,
    symbol_errors,
    yield_breakdown,
)
from prism_share.codec.decoder import ALL_DECODERS, DecoderOptions, read_index_band, read_symbols
from prism_share.codec.framing import frame_capacity
from prism_share.codec.layout import band_cell_cost
from prism_share.codec.params import (
    ASSUMED_FPS,
    INDEX_BAND_REFERENCE,
    PIXEL_SOURCES,
    RECTIFY_DEFAULT_KERNEL,
    UNDETECTED_CW_ERRORS,
    CodecParams,
)
from prism_share.ingest.ingest import (
    IngestedRun,
    detection_plane,
    ingest_run,
    load_planes,
    pixel_source,
    planes_needed,
)
from prism_share.ingest.manifest import RunRejected
from prism_share.ingest.rundef import DEFAULT_RUNS_DIR

FloatArray = npt.NDArray[np.float64]
REFERENCE, CODE, UNKNOWN = "reference", "code", "unknown"


OUT_OF_SAMPLE, IN_SAMPLE = "out_of_sample", "in_sample"
RS_SELECTIONS = (OUT_OF_SAMPLE, IN_SAMPLE)
IN_SAMPLE_WARNING = (
    "IN-SAMPLE RS SELECTION: the RS code is chosen on the same frames its goodput is scored on. Every best_* goodput "
    "is optimistic, most of all near the yield cliff, and must not be reported as a result."
)


class InSampleWarning(UserWarning):
    """Goodput is being computed with the RS code chosen on the frames it is scored on."""


@dataclass(frozen=True)
class SweepConfig:
    sources: tuple[str, ...] = PIXEL_SOURCES
    # Headline first (pre-registered, README 8.1); the other three are sensitivity analysis.
    decoders: tuple[DecoderOptions, ...] = ALL_DECODERS
    kernel: str = RECTIFY_DEFAULT_KERNEL
    flat_field: bool = False
    min_source_px_per_cell: float | None = None
    ecc_lengths: tuple[int, ...] = (155, 255)
    overlays_per_run: int = 3
    #: "out_of_sample" (default) or "in_sample" (biased; warns). See the module docstring.
    rs_selection: str = OUT_OF_SAMPLE

    def __post_init__(self) -> None:
        if self.rs_selection not in RS_SELECTIONS:
            raise ValueError(f"rs_selection must be one of {RS_SELECTIONS}, got {self.rs_selection!r}")
        if self.rs_selection == IN_SAMPLE:
            warnings.warn(IN_SAMPLE_WARNING, InSampleWarning, stacklevel=2)


@dataclass
class RunResult:
    run_id: str
    frames: pd.DataFrame
    overlays: list[Path] = field(default_factory=list)


def _sources_for(run: IngestedRun, sources: tuple[str, ...]) -> tuple[str, ...]:
    """The requested sources this run can actually provide.

    A single luma plane carries no colour, so 'y' serves monochrome codes only;
    the JPEG path has exactly one source whatever was requested.
    """
    available = run.available_sources
    if run.is_jpeg:
        return available
    depth = run.definition.params.colour_depth
    return tuple(s for s in sources if s in available and (s != "y" or depth == 1))


def _ecc_lengths(params: CodecParams, cfg: SweepConfig) -> list[int]:
    """Codeword lengths evaluated: the configured ones plus the run's own RS length (for the fixed-RS columns)."""
    return sorted(set(cfg.ecc_lengths) | {params.ecc_total})


def process_run(run: IngestedRun, cfg: SweepConfig, *, front_end: FrontEnd = locate_fiducials,
                overlay_dir: Path | None = None) -> RunResult:
    params = run.definition.params
    sources = _sources_for(run, cfg.sources)
    base_run = {**run.manifest.columns(), **run.definition.condition_columns(),
                **{f"codec_{k}": v for k, v in params.to_dict().items()},
                "colour_depth": params.colour_depth, "cell_px": params.cell_px, "params_label": params.label}
    lengths = _ecc_lengths(params, cfg)
    by_index = run.definition.by_index
    result = RunResult(run.manifest.run_id, pd.DataFrame())

    # Pass 1: detect every capture once and read the frame index it carries.
    detections: dict[str, tuple[Detection, str, dict[str, Any]]] = {}
    successes = 0
    for record in run.frames:
        y = detection_plane(run, record)
        det = detect(y, params, kernel=cfg.kernel, front_end=front_end)
        kind, band = UNKNOWN, {"band_index": None, "band_agreement": np.nan, "band_margin": np.nan}
        if det.record.detected:
            readout = read_index_band(det.rectified, params)
            band = {"band_index": readout.index, "band_agreement": readout.agreement, "band_margin": readout.margin}
            kind = REFERENCE if readout.index == INDEX_BAND_REFERENCE else CODE
        detections[record.file_stem] = (Detection(det.record, None, det.refined_corners), kind, band)
        if overlay_dir is not None and (not det.record.detected or successes < cfg.overlays_per_run):
            path = overlay_dir / run.manifest.run_id / f"{record.file_stem}_overlay.png"
            write_overlay(path, y, det, params)
            result.overlays.append(path)
        successes += int(det.record.detected)

    flat: dict[str, FlatField] = {}
    if cfg.flat_field:
        ref_records = [r for r in run.frames if detections[r.file_stem][1] == REFERENCE]
        if ref_records:
            first = ref_records[0]
            planes = load_planes(run, first, planes_needed(sources))
            rec = detections[first.file_stem][0].record
            flat = {s: FlatField.from_reference(rectify_with(pixel_source(run, planes, s), rec, params), params) for s in sources}

    # Pass 2: decode code captures from every source with every decoder.
    rows: list[dict[str, Any]] = []
    for record in run.frames:
        det, kind, band = detections[record.file_stem]
        rec = det.record
        base = {**base_run, **record.columns(), **rec.to_row(), "stimulus_kind": kind, **band}
        below = (rec.detected and cfg.min_source_px_per_cell is not None
                 and rec.source_px_per_cell is not None and rec.source_px_per_cell < cfg.min_source_px_per_cell)
        excluded = REFERENCE if kind == REFERENCE else ("below_resolution" if below else None)
        planes = load_planes(run, record, planes_needed(sources)) if (rec.detected and excluded is None) else None
        for source in sources:
            rectified = None
            if planes is not None:
                rectified = rectify_with(pixel_source(run, planes, source), rec, params, flat.get(source))
            for options in cfg.decoders:
                row = {**base, "pixel_source": source, "decoder": options.label, "excluded": excluded,
                       "flat_field_applied": source in flat, "stimulus_block_id": None, "identified": False,
                       "shape_ser": np.nan, "colour_ser": np.nan, "symbol_ser": np.nan, "byte_error_rate": np.nan}
                worst = {n: UNDETECTED_CW_ERRORS for n in lengths}
                if rectified is not None:
                    truth = by_index.get(band["band_index"])
                    if truth is not None:
                        readout = read_symbols(rectified, params, options)
                        errs = symbol_errors(truth.glyphs, truth.colours, readout.glyphs, readout.colours)
                        berr = byte_errors(truth.glyphs, truth.colours, readout.glyphs, readout.colours, params)
                        worst = {n: ecc_sim.max_codeword_errors(berr, n) for n in lengths}
                        row.update(stimulus_block_id=truth.block_id, identified=True, shape_ser=errs.shape_ser,
                                   colour_ser=errs.colour_ser, symbol_ser=errs.symbol_ser,
                                   byte_error_rate=float(berr.mean()))
                row.update({f"max_cw_errors_n{n}": v for n, v in worst.items()})
                rows.append(row)
    result.frames = pd.DataFrame(rows)
    return result


def summarise(frames: pd.DataFrame, cfg: SweepConfig) -> pd.DataFrame:
    """One row per (run, pixel source, decoder); the run fixes configuration, condition and device."""
    out = []
    provenance = ["device_manufacturer", "device_model", "capture_width", "capture_height", "yuv_matrix", "yuv_range",
                  "yuv_range_source", "control_protocol", "app_version", "params_label", "colour_depth", "cell_px"]
    condition_cols = [c for c in frames.columns if c.startswith("condition_")]
    for (run_id, source, decoder), g in frames.groupby(["run_id", "pixel_source", "decoder"], sort=True):
        first = g.iloc[0]
        params = CodecParams.from_dict({c[len("codec_"):]: int(first[c]) for c in g.columns if c.startswith("codec_")})
        code = g[g["excluded"].isna()]
        detected = code["detected"].astype(bool).to_numpy()
        identified = code[code["identified"].astype(bool)]
        row: dict[str, Any] = {
            "run_id": run_id, "pixel_source": source, "decoder": decoder,
            **{c: first[c] for c in provenance + condition_cols},
            "kernel": cfg.kernel, "flat_field": bool(g["flat_field_applied"].any()),
            "min_source_px_per_cell": cfg.min_source_px_per_cell,
            "n_captures": len(g), "n_reference": int((g["excluded"] == REFERENCE).sum()),
            "n_below_resolution": int((g["excluded"] == "below_resolution").sum()),
            "n_code_frames": len(code), "n_identified": len(identified),
            "n_band_index_unknown": int((code["detected"].astype(bool) & ~code["identified"].astype(bool)).sum()),
            "band_agreement_min": code["band_agreement"].min(),
            "shape_ser": identified["shape_ser"].mean(), "colour_ser": identified["colour_ser"].mean(),
            "symbol_ser": identified["symbol_ser"].mean(),
            "source_px_per_cell_median": code["source_px_per_cell"].median(),
            "source_px_per_cell_min": code["source_px_per_cell_min"].min(),
            "reprojection_error_px_median": code["reprojection_error_px"].median(),
            "band_cells_lost": band_cell_cost(params),
            "band_cell_cost_pct": 100 * band_cell_cost(params) / (frame_capacity(params).n_cells + band_cell_cost(params)),
        }
        # Split by capture position (fixed before analysis); in-sample only if explicitly asked for.
        select = ecc_sim.selection_mask(code["frame_index"].to_numpy())  # frames.jsonl "index": capture position
        if cfg.rs_selection == IN_SAMPLE:
            select = np.ones(len(code), dtype=bool)
            evaluation = select
        else:
            evaluation = ~select
        row.update(rs_selection=cfg.rs_selection, n_selection_frames=int(select.sum()),
                   n_evaluation_frames=int(evaluation.sum()))
        evaluated = code[evaluation]
        eval_detected = detected[evaluation]
        # Fixed RS: the configuration's own (ecc_total, ecc_data), counted on the evaluation frames.
        n, k = params.ecc_total, params.ecc_data
        if len(evaluated):
            fixed = yield_breakdown([frame_outcome(d, int(w), n, k)
                                     for d, w in zip(eval_detected, evaluated[f"max_cw_errors_n{n}"], strict=True)])
            row.update(_outcome_columns("fixed", fixed, params, n, k))
        # Best RS: chosen on the selection frames, counted on the evaluation frames (detection failures never recover).
        if select.any() and len(evaluated):
            chosen = ecc_sim.best_code({m: code.loc[select, f"max_cw_errors_n{m}"].tolist() for m in _ecc_lengths(params, cfg)},
                                       params)
            bd = yield_breakdown([frame_outcome(d, int(w), chosen.n, chosen.k)
                                  for d, w in zip(eval_detected, evaluated[f"max_cw_errors_n{chosen.n}"], strict=True)])
            row.update(_outcome_columns("best", bd, params, chosen.n, chosen.k))
        else:  # no held-out or no selection frames: no best-code goodput at all
            row.update({c: np.nan for c in _outcome_column_names("best")})
        out.append(row)
    return pd.DataFrame(out)


def _outcome_column_names(prefix: str) -> list[str]:
    return [f"{prefix}_{c}" for c in ("rs", "n_detection_failed", "n_not_recoverable", "n_recovered", "yield", "payload_bytes",
                                      "payload_bytes_band_credited", "goodput_mbps", "goodput_band_credited_mbps")]


def _outcome_columns(prefix: str, bd: Any, params: CodecParams, n: int, k: int) -> dict[str, Any]:
    payload = ecc_sim.payload_bytes_per_frame(params, n, k)
    credited = ecc_sim.payload_bytes_per_frame(params, n, k, band_credited=True)
    return {
        f"{prefix}_rs": f"RS({n},{k})",
        f"{prefix}_n_detection_failed": bd.detection_failed,
        f"{prefix}_n_not_recoverable": bd.not_recoverable,
        f"{prefix}_n_recovered": bd.recovered,
        f"{prefix}_yield": bd.frame_yield,
        f"{prefix}_payload_bytes": payload,
        f"{prefix}_payload_bytes_band_credited": credited,
        f"{prefix}_goodput_mbps": goodput_mbit_per_s(payload, bd.frame_yield, ASSUMED_FPS),
        f"{prefix}_goodput_band_credited_mbps": goodput_mbit_per_s(credited, bd.frame_yield, ASSUMED_FPS),
    }


GOODPUT_COLUMNS = {"measured": "goodput_mbps", "band_credited": "goodput_band_credited_mbps"}


def winners_both_columns(summary: pd.DataFrame, rs: str = "best") -> pd.DataFrame:
    """The goodput-maximising configuration per situation, under each goodput column.

    A situation is (device, condition, pixel source, decoder). ``same_winner``
    False means the index band's cell cost changes which configuration wins,
    i.e. apparatus, not physics, would be deciding the result there.
    """
    keys = ["device_model", *sorted(c for c in summary.columns if c.startswith("condition_")), "pixel_source", "decoder"]
    rows = []
    for key, g in summary.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        for label, column in GOODPUT_COLUMNS.items():
            top = g.loc[g[f"{rs}_{column}"].idxmax()]
            record.update({f"winner_{label}": f"{int(top['colour_depth'])} colours, {int(top['cell_px'])} px",
                           f"winner_{label}_mbps": top[f"{rs}_{column}"], f"winner_{label}_run_id": top["run_id"]})
        record["same_winner"] = record["winner_measured"] == record["winner_band_credited"]
        record["n_runs"] = len(g)
        rows.append(record)
    return pd.DataFrame(rows)


def sweep(run_dirs: list[Path], cfg: SweepConfig, *, runs_dir: Path = DEFAULT_RUNS_DIR, front_end: FrontEnd = locate_fiducials,
          overlay_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Process many runs. Rejected runs are reported and skipped, never half-processed."""
    frames, rejected = [], {}
    for run_dir in run_dirs:
        try:
            run = ingest_run(run_dir, runs_dir)
        except RunRejected as exc:
            rejected[str(run_dir)] = str(exc)
            continue
        frames.append(process_run(run, cfg, front_end=front_end, overlay_dir=overlay_dir).frames)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarise(table, cfg) if len(table) else pd.DataFrame()
    return table, summary, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--out", default="data/sweep")
    parser.add_argument("--figures", default=None, help="write provenance-stamped figures here")
    parser.add_argument("--sources", default=",".join(PIXEL_SOURCES))
    parser.add_argument("--kernel", default=RECTIFY_DEFAULT_KERNEL)
    parser.add_argument("--flat-field", action="store_true")
    parser.add_argument("--min-px-per-cell", type=float, default=None)
    parser.add_argument("--in-sample", action="store_true",
                        help="choose the RS code on the frames it is scored on (biased; for measuring the bias only)")
    args = parser.parse_args(argv)
    if args.in_sample:
        print(f"WARNING: {IN_SAMPLE_WARNING}", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InSampleWarning)  # already printed above
        cfg = SweepConfig(sources=tuple(args.sources.split(",")), kernel=args.kernel, flat_field=args.flat_field,
                          min_source_px_per_cell=args.min_px_per_cell,
                          rs_selection=IN_SAMPLE if args.in_sample else OUT_OF_SAMPLE)
    out = Path(args.out)
    try:
        frames, summary, rejected = sweep([Path(d) for d in args.run_dirs], cfg, runs_dir=Path(args.runs_dir),
                                          overlay_dir=out / "overlays")
    except NotImplementedError as exc:
        print(f"cannot process captures yet: {exc}", file=sys.stderr)
        return 3
    for run_dir, reason in rejected.items():
        print(f"REJECTED {run_dir}: {reason}", file=sys.stderr)
    if summary.empty:
        print("no runs processed", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(out / "frames.parquet", index=False)
    summary.to_parquet(out / "summary.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    both = winners_both_columns(summary)
    both.to_csv(out / "winners.csv", index=False, float_format="%.6g")
    print(f"wrote {out}/frames.parquet ({len(frames)} rows) and summary.parquet ({len(summary)} rows)")
    differ = both[~both["same_winner"]]
    if len(differ):
        print(f"WARNING: the index band's cell cost changes the winner in {len(differ)} situation(s); see {out}/winners.csv",
              file=sys.stderr)
    else:
        print("winner identical with and without the band's cells credited back, in every situation")
    if args.figures:
        from prism_share.analysis import plots

        for path in plots.captured_figures(summary, Path(args.figures)):
            print(f"figure: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
