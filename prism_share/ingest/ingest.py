"""Ingest a pulled run folder: enforce the contract's six rules, load frames, merge with detection.

    python -m prism_share.ingest.ingest data/<run_id>           # validate and report
    python -m prism_share.ingest.ingest data/<run_id> --detect  # also detect every kept frame (needs the front end)

The six rules of docs/capture-format.md, where each is enforced:

1. unknown ``schema_version``           -> ``RunRejected``  (manifest.read_manifest)
2. no laptop-side run definition        -> ``RunRejected``  (rundef.load_run_definition)
3. drop every tainted frame, count them and count each taint reason (IngestReport)
4. frames.tainted / frames.written > 1 % -> ``RunRejected``, "must be repeated"
5. frames.written != frames.requested   -> warning (IngestReport.warnings, printed)
6. resolution, YUV matrix, YUV range come from the manifest; every plane loaded
   is checked against the manifest resolution; unknown values reject the run.

Beyond the six rules, ingest also refuses a run whose files contradict each
other: frames.jsonl must have ``frames.written`` lines and exactly
``frames.tainted`` tainted ones, since rule 4 is computed from those counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image

from prism_share.analysis.detect import FrontEnd, detect, locate_fiducials, write_overlay
from prism_share.codec.params import JPEG_PIXEL_SOURCE, PIXEL_SOURCES, RECTIFY_DEFAULT_KERNEL
from prism_share.colourspace import YuvSpec, upsample_chroma, yuv_to_rgb
from prism_share.ingest.manifest import (
    JPEG_PIXEL_FORMAT,
    MAX_TAINTED_FRACTION,
    PIXEL_FORMATS,
    TAINT_REASONS,
    YUV_PIXEL_FORMAT,
    ContractViolation,
    FrameRecord,
    Manifest,
    RunRejected,
    read_frames,
    read_manifest,
)
from prism_share.ingest.rundef import DEFAULT_RUNS_DIR, RunDefinition, load_run_definition

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IngestReport:
    run_id: str
    frames_requested: int
    frames_written: int
    frames_tainted: int
    tainted_fraction: float
    dropped: int
    dropped_by_reason: dict[str, int]
    kept: int
    warnings: tuple[str, ...]
    yuv_range_source: str

    def text(self) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in self.dropped_by_reason.items() if v) or "none"
        lines = [
            f"run {self.run_id}: {self.kept} frames kept, {self.dropped} tainted frames dropped "
            f"({100 * self.tainted_fraction:.2f} % of {self.frames_written} written; by reason: {reasons})",
            f"YUV range source: {self.yuv_range_source}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass(frozen=True)
class IngestedRun:
    run_dir: Path
    manifest: Manifest
    definition: RunDefinition
    frames: tuple[FrameRecord, ...]
    """Untainted frames only, in frames.jsonl order."""
    report: IngestReport

    @property
    def yuv_spec(self) -> YuvSpec:
        return self.manifest.yuv_spec

    @property
    def is_jpeg(self) -> bool:
        """True on the JPEG comparison path: one image per frame, one pixel source."""
        return self.manifest.pixel_format == JPEG_PIXEL_FORMAT

    @property
    def available_sources(self) -> tuple[str, ...]:
        """Pixel sources this run's files can provide."""
        return (JPEG_PIXEL_SOURCE,) if self.is_jpeg else PIXEL_SOURCES


def ingest_run(run_dir: str | Path, runs_dir: Path = DEFAULT_RUNS_DIR) -> IngestedRun:
    """Validate a pulled run folder and return its untainted frames. Raises RunRejected."""
    folder = Path(run_dir)
    manifest = read_manifest(folder / "manifest.json")  # rule 1
    definition = load_run_definition(manifest.run_id, runs_dir)  # rule 2

    if manifest.pixel_format not in PIXEL_FORMATS:  # rule 6
        raise RunRejected(f"run {manifest.run_id}: pixel_format {manifest.pixel_format!r} is not one this ingest "
                          f"can read ({list(PIXEL_FORMATS)})")
    # Rule 6: resolution, matrix and range must be readable now, not discovered mid-run.
    # A JPEG run carries no planes, so its Y'CbCr fields are not consulted.
    _ = manifest.capture_resolution
    if manifest.pixel_format == YUV_PIXEL_FORMAT:
        _ = (manifest.yuv_spec, manifest.yuv_range_source)

    written, tainted, requested = manifest.frames_written, manifest.frames_tainted, manifest.frames_requested
    if written <= 0:
        raise RunRejected(f"run {manifest.run_id}: frames.written is {written}; nothing to ingest. Repeat the run.")
    fraction = tainted / written
    if fraction > MAX_TAINTED_FRACTION:  # rule 4
        raise RunRejected(
            f"run {manifest.run_id}: {tainted} of {written} frames are tainted ({100 * fraction:.2f} % > "
            f"{100 * MAX_TAINTED_FRACTION:g} %). The camera lock failed during this run; it is not partially "
            "usable and must be repeated."
        )

    records = read_frames(folder / "frames.jsonl")
    if len(records) != written:
        raise ContractViolation(f"run {manifest.run_id}: frames.jsonl has {len(records)} lines but frames.written is {written}")
    tainted_records = [r for r in records if r.tainted]
    if len(tainted_records) != tainted:
        raise ContractViolation(f"run {manifest.run_id}: frames.jsonl marks {len(tainted_records)} frames tainted "
                                f"but frames.tainted is {tainted}")

    by_reason = Counter(reason for r in tainted_records for reason in r.taint_reasons)  # rule 3
    kept = tuple(r for r in records if not r.tainted)
    warnings = []
    if written != requested:  # rule 5
        warnings.append(f"frames.written ({written}) != frames.requested ({requested})")
    report = IngestReport(
        run_id=manifest.run_id,
        frames_requested=requested,
        frames_written=written,
        frames_tainted=tainted,
        tainted_fraction=fraction,
        dropped=len(tainted_records),
        dropped_by_reason={reason: by_reason.get(reason, 0) for reason in TAINT_REASONS},
        kept=len(kept),
        warnings=tuple(warnings),
        yuv_range_source=manifest.yuv_range_source,
    )
    for w in warnings:
        print(f"WARNING ({manifest.run_id}): {w}", file=sys.stderr)
    return IngestedRun(folder, manifest, definition, kept, report)


# --------------------------------------------------------------------------- #
# Loading frames
# --------------------------------------------------------------------------- #


def _load_plane(path: Path, expected_hw: tuple[int, int], mode: str) -> npt.NDArray[np.uint8]:
    if not path.exists():
        raise ContractViolation(f"{path} is missing")
    with Image.open(path) as im:
        if im.mode != mode:
            raise ContractViolation(f"{path}: expected an 8-bit {'greyscale' if mode == 'L' else 'RGB'} PNG, got mode {im.mode}")
        arr = np.array(im, dtype=np.uint8)
    if arr.shape[:2] != expected_hw:
        raise ContractViolation(f"{path}: {arr.shape[1]}x{arr.shape[0]} does not match the manifest's "
                                f"capture_resolution ({expected_hw[1]}x{expected_hw[0]})")
    return arr


def load_planes(run: IngestedRun, record: FrameRecord, planes: tuple[str, ...] | None = None) -> dict[str, npt.NDArray[np.uint8]]:
    """Load a frame's files, each checked against the manifest resolution (rule 6)."""
    width, height = run.manifest.capture_resolution
    chroma_hw = (-(-height // 2), -(-width // 2))  # native 4:2:0 planes, never upsampled on the phone
    shapes = {"y": ((height, width), "L"), "u": (chroma_hw, "L"), "v": (chroma_hw, "L"),
              "rgb": ((height, width), "RGB"), "jpeg": ((height, width), "RGB")}
    if planes is None:
        planes = ("jpeg",) if run.is_jpeg else ("y", "u", "v", "rgb")
    return {p: _load_plane(record.path(run.run_dir, p), *shapes[p]) for p in planes}


def detection_plane(run: IngestedRun, record: FrameRecord) -> npt.NDArray[np.uint8]:
    """The image detection runs on: the Y plane, or the JPEG image on that path."""
    plane = "jpeg" if run.is_jpeg else "y"
    return load_planes(run, record, (plane,))[plane]


def pixel_source(run: IngestedRun, planes: dict[str, npt.NDArray[np.uint8]], source: str) -> FloatArray:
    """The captured frame as decoded from one pixel source (params.PIXEL_SOURCES)."""
    if source in ("y", "jpeg"):
        return planes[source].astype(np.float64)
    if source == "rgb":
        return planes["rgb"].astype(np.float64)
    if source in ("yuv_nearest", "yuv_bilinear"):
        method = source.split("_", 1)[1]
        y = planes["y"]
        cb = upsample_chroma(planes["u"], y.shape, method)  # type: ignore[arg-type]
        cr = upsample_chroma(planes["v"], y.shape, method)  # type: ignore[arg-type]
        return yuv_to_rgb(y, cb, cr, run.yuv_spec)
    raise ValueError(f"unknown pixel source {source!r}")


def planes_needed(sources: tuple[str, ...]) -> tuple[str, ...]:
    if "jpeg" in sources:
        return ("jpeg",)
    need = {"y"}  # detection always runs on the Y plane
    for s in sources:
        need |= {"rgb"} if s == "rgb" else ({"u", "v"} if s.startswith("yuv") else set())
    return tuple(p for p in ("y", "u", "v", "rgb") if p in need)


# --------------------------------------------------------------------------- #
# Detection + merge
# --------------------------------------------------------------------------- #


def detect_run(
    run: IngestedRun,
    *,
    front_end: FrontEnd = locate_fiducials,
    kernel: str = RECTIFY_DEFAULT_KERNEL,
    overlay_dir: Path | None = None,
    overlays_per_run: int = 3,
) -> pd.DataFrame:
    """Detect every kept frame (on its Y plane) and merge phone and detection records into one table.

    One row per kept frame: manifest provenance, the phone's per-frame record,
    then the detection record. Detection failure is a row with detected=False
    and a failure_reason, never an exception. Overlays are written for every
    failure and the first ``overlays_per_run`` successes.
    """
    rows = []
    successes = 0
    for record in run.frames:
        y = detection_plane(run, record)
        detection = detect(y, run.definition.params, kernel=kernel, front_end=front_end)
        rec = detection.record
        rows.append({**run.manifest.columns(), **run.definition.condition_columns(), **record.columns(), **rec.to_row()})
        if overlay_dir is not None and (not rec.detected or successes < overlays_per_run):
            write_overlay(overlay_dir / f"{record.file_stem}_overlay.png", y, detection, run.definition.params)
        successes += int(rec.detected)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="laptop-side run definitions")
    parser.add_argument("--detect", action="store_true", help="also detect every kept frame and write the merged table")
    parser.add_argument("--out", default="data/processed", help="where merged tables and overlays go")
    args = parser.parse_args(argv)
    try:
        run = ingest_run(args.run_dir, Path(args.runs_dir))
    except RunRejected as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2
    print(run.report.text())
    out = Path(args.out) / run.manifest.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "ingest_report.json").write_text(json.dumps(asdict(run.report), indent=1) + "\n")
    if args.detect:
        table = detect_run(run, overlay_dir=out / "overlays")
        table.to_parquet(out / "frames_detected.parquet", index=False)
        print(f"detected {int(table['detected'].sum())} of {len(table)} kept frames; wrote {out}/frames_detected.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
