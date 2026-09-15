"""Reading and validating the phone's run files, exactly as docs/capture-format.md defines them.

This module is the only place that encodes the capture contract (schema
version 1). It adds no fields: every name below appears in the contract.
Where the contract leaves something open, this module refuses rather than
guesses:

* ``color.yuv_matrix`` / ``color.yuv_range`` are not enumerated by the
  contract. Only values whose meaning is unambiguous are accepted ("BT601",
  "BT709"; "limited", "full"); anything else rejects the run (rule 6: never
  assume).
* ``camera.capture_resolution`` is read as [width, height] (Camera2 ``Size``
  order) and every plane loaded is checked against it, so a wrong reading
  cannot pass silently.
* ``color.pixel_format`` marks the path: "YUV_420_888" (Y, U, V and RGB PNGs)
  or "JPEG" (one ``frame_NNNN.jpg`` per frame, a comparison baseline that is
  never a headline measurement). Any other value rejects the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_share.colourspace import YuvSpec

# --------------------------------------------------------------------------- #
# Contract constants (docs/capture-format.md)
# --------------------------------------------------------------------------- #

#: Schema versions this ingest understands (rule 1).
SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
#: Where a run lives on the phone.
PHONE_CAPTURES_DIR = "/sdcard/Android/data/io.github.ahmedov.prismshare.capture/files/captures"
#: The closed set of taint reasons.
TAINT_REASONS: tuple[str, ...] = ("ae_state", "awb_state", "sensitivity_drift", "exposure_drift", "gains_drift")
#: Rule 4: refuse the run when frames.tainted / frames.written exceeds this.
MAX_TAINTED_FRACTION = 0.01
#: file_stem + suffix = the frame's files.
PLANE_SUFFIXES: dict[str, str] = {"y": "_y.png", "u": "_u.png", "v": "_v.png", "rgb": "_rgb.png", "jpeg": ".jpg"}
YUV_PIXEL_FORMAT = "YUV_420_888"
JPEG_PIXEL_FORMAT = "JPEG"
PIXEL_FORMATS = (YUV_PIXEL_FORMAT, JPEG_PIXEL_FORMAT)
_MATRICES = {"BT601": "bt601", "BT709": "bt709"}
_RANGES = {"limited": False, "full": True}
_RANGE_SOURCES = ("read", "assumed")


class RunRejected(Exception):
    """Ingest refuses the run. The message says why and what to do about it."""


class ContractViolation(RunRejected):
    """The run's files do not conform to docs/capture-format.md."""


def _get(doc: dict[str, Any], dotted: str, where: str) -> Any:
    node: Any = doc
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ContractViolation(f"{where}: required field '{dotted}' is missing")
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
# manifest.json
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Manifest:
    """manifest.json. ``raw`` keeps every field; properties expose the ones ingest uses."""

    raw: dict[str, Any]
    path: Path

    def field(self, dotted: str) -> Any:
        return _get(self.raw, dotted, str(self.path))

    @property
    def schema_version(self) -> int:
        return int(self.field("schema_version"))

    @property
    def run_id(self) -> str:
        return str(self.field("run_id"))

    @property
    def device_model(self) -> str:
        return str(self.field("device.model"))

    @property
    def device_manufacturer(self) -> str:
        return str(self.field("device.manufacturer"))

    @property
    def capture_resolution(self) -> tuple[int, int]:
        """(width, height) in pixels, from camera.capture_resolution."""
        value = self.field("camera.capture_resolution")
        if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and v > 0 for v in value)):
            raise ContractViolation(f"{self.path}: camera.capture_resolution must be [width, height], got {value!r}")
        return int(value[0]), int(value[1])

    @property
    def pixel_format(self) -> str:
        return str(self.field("color.pixel_format"))

    @property
    def yuv_spec(self) -> YuvSpec:
        """The Y'CbCr matrix and range from the manifest (rule 6)."""
        matrix, rng = self.field("color.yuv_matrix"), self.field("color.yuv_range")
        if matrix not in _MATRICES:
            raise RunRejected(f"{self.path}: color.yuv_matrix {matrix!r} is not one this ingest can interpret "
                              f"({sorted(_MATRICES)}); refusing rather than assuming one")
        if rng not in _RANGES:
            raise RunRejected(f"{self.path}: color.yuv_range {rng!r} is not one this ingest can interpret "
                              f"({sorted(_RANGES)}); refusing rather than assuming one")
        return YuvSpec(matrix=_MATRICES[matrix], full_range=_RANGES[rng])

    @property
    def yuv_range_source(self) -> str:
        value = self.field("color.yuv_range_source")
        if value not in _RANGE_SOURCES:
            raise ContractViolation(f"{self.path}: color.yuv_range_source must be one of {_RANGE_SOURCES}, got {value!r}")
        return str(value)

    @property
    def frames_requested(self) -> int:
        return int(self.field("frames.requested"))

    @property
    def frames_written(self) -> int:
        return int(self.field("frames.written"))

    @property
    def frames_tainted(self) -> int:
        return int(self.field("frames.tainted"))

    def columns(self) -> dict[str, Any]:
        """The manifest facts every analysis row carries (provenance)."""
        width, height = self.capture_resolution
        return {
            "run_id": self.run_id,
            "captured_at": self.raw.get("captured_at"),
            "device_manufacturer": self.device_manufacturer,
            "device_model": self.device_model,
            "android_release": self.raw.get("device", {}).get("android_release"),
            "camera_id": self.raw.get("camera", {}).get("camera_id"),
            "hardware_level": self.raw.get("camera", {}).get("hardware_level"),
            "capture_width": width,
            "capture_height": height,
            "control_protocol": self.raw.get("control", {}).get("protocol"),
            "reference_frame_id": self.raw.get("control", {}).get("reference_frame_id"),
            "pixel_format": self.pixel_format,
            "yuv_matrix": self.field("color.yuv_matrix"),
            "yuv_range": self.field("color.yuv_range"),
            "yuv_range_source": self.yuv_range_source,
            "app_version": self.raw.get("app", {}).get("version"),
            "app_git_commit": self.raw.get("app", {}).get("git_commit"),
        }


def read_manifest(path: str | Path) -> Manifest:
    """Parse manifest.json and apply rule 1 (unknown schema_version rejects the run)."""
    p = Path(path)
    if not p.exists():
        raise ContractViolation(f"{p} not found: a run folder must contain manifest.json")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{p}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ContractViolation(f"{p}: must contain one JSON object")
    manifest = Manifest(raw=raw, path=p)
    version = manifest.field("schema_version")
    if version not in SCHEMA_VERSIONS:
        raise RunRejected(f"{p}: schema_version {version!r} is unknown (this ingest reads {sorted(SCHEMA_VERSIONS)})")
    return manifest


# --------------------------------------------------------------------------- #
# frames.jsonl
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrameRecord:
    """One line of frames.jsonl."""

    index: int
    file_stem: str
    sensor_timestamp_ns: int
    ae_state: str
    awb_state: str
    af_state: str
    sensor_sensitivity: int
    sensor_exposure_time_ns: int
    color_correction_gains: tuple[float, ...]
    rolling_shutter_skew_ns: int
    tainted: bool
    taint_reasons: tuple[str, ...]

    def path(self, run_dir: Path, plane: str) -> Path:
        """The file for ``plane`` ('y', 'u', 'v', 'rgb'): file_stem + suffix, never rebuilt from index."""
        return run_dir / f"{self.file_stem}{PLANE_SUFFIXES[plane]}"

    def columns(self) -> dict[str, Any]:
        gains = self.color_correction_gains
        return {
            "frame_index": self.index,
            "file_stem": self.file_stem,
            "sensor_timestamp_ns": self.sensor_timestamp_ns,
            "ae_state": self.ae_state,
            "awb_state": self.awb_state,
            "af_state": self.af_state,
            "sensor_sensitivity": self.sensor_sensitivity,
            "sensor_exposure_time_ns": self.sensor_exposure_time_ns,
            **{f"color_correction_gain_{i}": g for i, g in enumerate(gains)},
            "rolling_shutter_skew_ns": self.rolling_shutter_skew_ns,
            "tainted": self.tainted,
            "taint_reasons": ",".join(self.taint_reasons),
        }


_FRAME_FIELDS: dict[str, type | tuple[type, ...]] = {
    "index": int,
    "file_stem": str,
    "sensor_timestamp_ns": int,
    "ae_state": str,
    "awb_state": str,
    "af_state": str,
    "sensor_sensitivity": int,
    "sensor_exposure_time_ns": int,
    "color_correction_gains": list,
    "rolling_shutter_skew_ns": int,
    "tainted": bool,
    "taint_reasons": list,
}


def parse_frame_record(obj: Any, where: str) -> FrameRecord:
    if not isinstance(obj, dict):
        raise ContractViolation(f"{where}: each line must be one JSON object")
    for key, kind in _FRAME_FIELDS.items():
        if key not in obj:
            raise ContractViolation(f"{where}: required field '{key}' is missing")
        value = obj[key]
        if isinstance(value, bool) and kind is int:
            raise ContractViolation(f"{where}: '{key}' must be an integer, got {value!r}")
        if not isinstance(value, kind):
            raise ContractViolation(f"{where}: '{key}' has the wrong type ({type(value).__name__})")
    reasons = tuple(obj["taint_reasons"])
    unknown = [r for r in reasons if r not in TAINT_REASONS]
    if unknown:
        raise ContractViolation(f"{where}: taint_reasons {unknown} are outside the closed set {TAINT_REASONS}")
    if obj["tainted"] != bool(reasons):
        raise ContractViolation(f"{where}: 'tainted' must be true if and only if 'taint_reasons' is non-empty")
    gains = obj["color_correction_gains"]
    if not all(isinstance(g, int | float) and not isinstance(g, bool) for g in gains):
        raise ContractViolation(f"{where}: 'color_correction_gains' must be numbers")
    return FrameRecord(
        index=obj["index"],
        file_stem=obj["file_stem"],
        sensor_timestamp_ns=obj["sensor_timestamp_ns"],
        ae_state=obj["ae_state"],
        awb_state=obj["awb_state"],
        af_state=obj["af_state"],
        sensor_sensitivity=obj["sensor_sensitivity"],
        sensor_exposure_time_ns=obj["sensor_exposure_time_ns"],
        color_correction_gains=tuple(float(g) for g in gains),
        rolling_shutter_skew_ns=obj["rolling_shutter_skew_ns"],
        tainted=obj["tainted"],
        taint_reasons=reasons,
    )


def read_frames(path: str | Path) -> list[FrameRecord]:
    p = Path(path)
    if not p.exists():
        raise ContractViolation(f"{p} not found: a run folder must contain frames.jsonl")
    records = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractViolation(f"{p}:{lineno}: not valid JSON ({exc})") from exc
        records.append(parse_frame_record(obj, f"{p}:{lineno}"))
    stems = [r.file_stem for r in records]
    if len(set(stems)) != len(stems):
        raise ContractViolation(f"{p}: duplicate file_stem values")
    return records
