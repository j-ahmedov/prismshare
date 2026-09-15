"""Synthetic run folders written exactly in the docs/capture-format.md layout.

Test-only. The *contents* are synthetic (code frames imaged through a known
homography, then encoded to 8-bit BT.601 limited-range Y'CbCr 4:2:0 as a
YUV_420_888 capture would be); the *format* is the contract's, field for field,
so ingest is exercised exactly as it will be on real runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image

from prism_share.analysis import detect as det
from prism_share.analysis.detect import FrontEndResult
from prism_share.codec.params import FIDUCIAL_IDS, CodecParams
from prism_share.colourspace import YuvSpec, rgb_to_yuv, subsample_chroma, upsample_chroma, yuv_to_rgb
from prism_share.ingest.rundef import write_clip_check_frames
from prism_share.transmit import display
from prism_share.transmit.reference import reference_frame

F = CodecParams().frame_px
FRAME_CORNERS = np.array([[-0.5, -0.5], [F - 0.5, -0.5], [F - 0.5, F - 0.5], [-0.5, F - 0.5]], np.float32)
#: A near-frontal view at ~1.3 capture px per screen px.
DEFAULT_QUAD = [[120, 90], [1450, 110], [1440, 1440], [100, 1425]]
DEFAULT_SIZE = (1560, 1530)  # (width, height)


def homography(quad: list[list[float]] = DEFAULT_QUAD) -> npt.NDArray[np.float64]:
    return cv2.getPerspectiveTransform(FRAME_CORNERS, np.array(quad, np.float32)).astype(np.float64)


def image_through(image: npt.NDArray, h: npt.NDArray[np.float64], size: tuple[int, int], ss: int = 2) -> npt.NDArray[np.float64]:
    s = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]])
    big = cv2.warpPerspective(np.asarray(image, np.float32), s @ h, (size[0] * ss, size[1] * ss),
                              flags=cv2.INTER_LINEAR, borderValue=(30, 30, 30))
    return np.clip(np.rint(cv2.resize(big, size, interpolation=cv2.INTER_AREA)), 0, 255).astype(np.float64)


def stub_front_end(h: npt.NDArray[np.float64], fail_stems: set[int] | None = None) -> det.FrontEnd:
    """Stands in for the unimplemented real front end: true corners, +/-1 px. Can be told to fail."""
    true = det.project(h, det.ideal_marker_corners(F))
    rng = np.random.default_rng(0)
    corners = {m: true[i] + rng.uniform(-1, 1, (4, 2)) for i, m in enumerate(FIDUCIAL_IDS)}
    calls = {"n": 0}

    def front_end(image: npt.NDArray, params: CodecParams) -> FrontEndResult:
        index = calls["n"]
        calls["n"] += 1
        if fail_stems and index in fail_stems:
            return FrontEndResult({}, failure_reason="stub told to fail")
        return FrontEndResult(dict(corners))

    return front_end


@dataclass
class RunSpec:
    run_id: str = "test_run"
    colour_depth: int = 4
    cell_px: int = 8
    n_code_frames: int = 6
    displayed_block: int = 0
    """The ONE frame this run holds on screen."""
    stray_captures: dict[int, int] = field(default_factory=dict)
    """Capture index -> a different block that capture shows instead (a protocol slip)."""
    reference_captures: tuple[int, ...] = ()
    """Capture indices (0-based, in order) that show the reference frame instead of a code."""
    tainted: dict[int, list[str]] = field(default_factory=dict)
    """Capture index -> taint reasons."""
    requested: int | None = None
    images: bool = True
    quad: list[list[float]] = field(default_factory=lambda: list(DEFAULT_QUAD))
    size: tuple[int, int] = DEFAULT_SIZE
    stem: str = "frame_{:04d}"
    pixel_format: str = "YUV_420_888"
    """"YUV_420_888" writes Y/U/V/RGB PNGs; "JPEG" writes one .jpg per frame."""
    manifest_overrides: dict[str, Any] = field(default_factory=dict)
    condition: dict[str, Any] = field(default_factory=lambda: {"distance_m": 0.5})
    write_definition: bool = True


@dataclass
class BuiltRun:
    run_dir: Path
    runs_dir: Path
    params: CodecParams
    h: npt.NDArray[np.float64]
    capture_blocks: list[int | None]
    """Per capture: block id actually shown, or None for a reference capture."""


def manifest_doc(spec: RunSpec, written: int, tainted: int) -> dict[str, Any]:
    """The contract's example manifest with the run-specific values filled in."""
    doc: dict[str, Any] = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "captured_at": "2026-09-14T16:22:31+02:00",
        "notes": "synthetic test run",
        "device": {"manufacturer": "samsung", "model": "SM-A346E", "android_release": "16", "api_level": 36,
                   "build_fingerprint": "samsung/a34xnsxx/a34x:16/..."},
        "camera": {"camera_id": "0", "hardware_level": "LIMITED", "capabilities": ["BACKWARD_COMPATIBLE", "BURST_CAPTURE"],
                   "active_array_size": [4000, 3000], "pixel_array_size": [4080, 3060], "color_filter_arrangement": "GRBG",
                   "capture_template": "TEMPLATE_STILL_CAPTURE", "capture_resolution": list(spec.size)},
        "control": {"protocol": "locked_auto", "reference_frame_id": "ref_v1_1024", "clip_low_pct": 0.02, "clip_high_pct": 0.0,
                    "frozen": {"sensor_sensitivity": 320, "sensor_exposure_time_ns": 16666667, "sensor_frame_duration_ns": 33333333,
                               "color_correction_gains": [1.94, 1.0, 1.0, 1.72], "color_correction_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               "lens_focus_distance": 2.44, "lens_aperture": 1.8, "ae_exposure_compensation": 0},
                    "applied_modes": {"tonemap": "FAST", "noise_reduction": "OFF", "edge": "OFF", "antibanding": "OFF",
                                      "video_stabilization": "OFF", "optical_stabilization": "OFF"}},
        "color": {"pixel_format": spec.pixel_format, "yuv_matrix": "BT601", "yuv_range": "limited",
                  "yuv_range_source": "assumed", "dataspace": None},
        "frames": {"requested": spec.requested if spec.requested is not None else written, "written": written, "tainted": tainted},
        "app": {"version": "0.1.0", "git_commit": "a1b2c3d"},
    }
    for dotted, value in spec.manifest_overrides.items():
        node = doc
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    return doc


def clip_check_block(root: Path) -> dict[str, Any]:
    """A real clip-check pair (frames written once per test root), in the run-definition format.

    Fixed to the two configurations the current luminance table names, so tests
    do not pay for recomputing it; test_rundef.py checks the generator itself.
    """
    out = {}
    for role, (depth, cell) in (("brightest", (1, 10)), ("darkest", (4, 4))):
        folder = write_clip_check_frames(CodecParams(colour_depth=depth, cell_px=cell), root / "clip_check")
        out[role] = {"colour_depth": depth, "cell_px": cell, "frames": str(folder),
                     "frame_mean_linear": 0.4375 if role == "brightest" else 0.2174,
                     "stops_vs_reference": 0.506 if role == "brightest" else -0.502}
    return out


def frame_line(index: int, stem: str, reasons: list[str]) -> dict[str, Any]:
    return {"index": index, "file_stem": stem, "sensor_timestamp_ns": 88123456789 + index * 33333333,
            "ae_state": "LOCKED", "awb_state": "LOCKED", "af_state": "FOCUSED_LOCKED", "sensor_sensitivity": 320,
            "sensor_exposure_time_ns": 16666667, "color_correction_gains": [1.94, 1.0, 1.0, 1.72],
            "rolling_shutter_skew_ns": 28000000, "tainted": bool(reasons), "taint_reasons": list(reasons)}


def build_run(root: Path, spec: RunSpec) -> BuiltRun:
    params = CodecParams(colour_depth=spec.colour_depth, cell_px=spec.cell_px)
    frames_dir = root / "displayed" / spec.run_id
    n_frames = max([spec.displayed_block, *spec.stray_captures.values()]) + 1
    display.main(["frames", "--colour-depth", str(params.colour_depth), "--cell-px", str(params.cell_px),
                  "--payload-bytes", "5000", "--n-frames", str(n_frames), "--out", str(frames_dir)])
    runs_dir = root / "experiments" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if spec.write_definition:
        (runs_dir / f"{spec.run_id}.yaml").write_text(yaml.safe_dump(
            {"run_id": spec.run_id, "frames": str(frames_dir), "displayed": [spec.displayed_block],
             "condition": spec.condition, "clip_check": clip_check_block(root)}))

    run_dir = root / "data" / spec.run_id
    run_dir.mkdir(parents=True)
    h = homography(spec.quad)
    n_total = spec.n_code_frames + len(spec.reference_captures)
    frames = {int(f.stem.split("_")[1]): np.array(Image.open(f)) for f in frames_dir.glob("frame_*.png")}
    ref = np.asarray(reference_frame(F))
    blocks: list[int | None] = []
    lines = []
    yuv = YuvSpec("bt601", False)
    for i in range(n_total):
        if i in spec.reference_captures:
            image, block = ref, None
        else:
            block = spec.stray_captures.get(i, spec.displayed_block)
            image = frames[block]
        blocks.append(block)
        stem = spec.stem.format(i)
        lines.append(frame_line(i, stem, spec.tainted.get(i, [])))
        if spec.images and spec.pixel_format == "JPEG":
            shot = image_through(image, h, spec.size)
            Image.fromarray(shot.astype(np.uint8)).save(run_dir / f"{stem}.jpg", quality=95)
        elif spec.images:
            shot = image_through(image, h, spec.size)
            y, cb, cr = rgb_to_yuv(shot, yuv)
            u, v = subsample_chroma(cb, 2), subsample_chroma(cr, 2)
            rgb = yuv_to_rgb(y, upsample_chroma(u, y.shape, "bilinear"), upsample_chroma(v, y.shape, "bilinear"), yuv)
            Image.fromarray(y).save(run_dir / f"{stem}_y.png")
            Image.fromarray(u).save(run_dir / f"{stem}_u.png")
            Image.fromarray(v).save(run_dir / f"{stem}_v.png")
            Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8)).save(run_dir / f"{stem}_rgb.png")
    tainted = sum(1 for line in lines if line["tainted"])
    (run_dir / "manifest.json").write_text(json.dumps(manifest_doc(spec, n_total, tainted), indent=1))
    (run_dir / "frames.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return BuiltRun(run_dir, runs_dir, params, h, blocks)
