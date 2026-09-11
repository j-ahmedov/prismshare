"""Colour-space conversions shared by the simulator, the transmitter and analysis.

* sRGB transfer functions (IEC 61966-2-1) and linear-light luminance, used to
  predict what a camera's auto-exposure meters.
* 8-bit Y'CbCr exactly as a YUV_420_888 capture stores it: a documented matrix
  (BT.601 by default, BT.709 selectable), limited or full range, every plane
  rounded to 8 bits, chroma planes subsampled. The same functions will convert
  real YUV captures in the ingest step, so simulation and measurement share one
  implementation.

All RGB arrays here are on a 0-255 scale (float or uint8), channel order RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from prism_share.codec.params import (
    LINEAR_LUMINANCE_WEIGHTS,
    YUV_C_OFFSET,
    YUV_DEFAULT_MATRIX,
    YUV_FULL_SPAN,
    YUV_LIMITED_C_SPAN,
    YUV_LIMITED_Y_OFFSET,
    YUV_LIMITED_Y_SPAN,
    YUV_MATRICES,
)

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]
Upsample = Literal["nearest", "bilinear"]

_FULL_SCALE = 255.0

# --------------------------------------------------------------------------- #
# sRGB and linear light
# --------------------------------------------------------------------------- #


def srgb_to_linear(encoded: npt.ArrayLike) -> FloatArray:
    """sRGB EOTF: encoded values in [0, 1] -> linear light in [0, 1]."""
    v = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: npt.ArrayLike) -> FloatArray:
    """Inverse sRGB EOTF: linear light in [0, 1] -> encoded values in [0, 1]."""
    v = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def linear_luminance(rgb: npt.ArrayLike) -> FloatArray:
    """Relative linear luminance (0-1) of 0-255 sRGB code values, per pixel."""
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64) / _FULL_SCALE)
    return lin @ np.array(LINEAR_LUMINANCE_WEIGHTS)


def mean_linear_luminance(rgb: npt.ArrayLike) -> float:
    """Average linear luminance of an image: what an averaging light meter reads."""
    return float(linear_luminance(rgb).mean())


# --------------------------------------------------------------------------- #
# Y'CbCr
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class YuvSpec:
    """How RGB is encoded into 8-bit Y'CbCr planes."""

    matrix: str = YUV_DEFAULT_MATRIX
    """Key of params.YUV_MATRICES: 'bt601' or 'bt709'."""
    full_range: bool = False
    """False = limited/video range (Y' 16-235, C 16-240); True = full/JFIF range."""

    def __post_init__(self) -> None:
        if self.matrix not in YUV_MATRICES:
            raise ValueError(f"unknown matrix {self.matrix!r}; choose from {sorted(YUV_MATRICES)}")

    @property
    def coefficients(self) -> tuple[float, float, float]:
        kr, kb = YUV_MATRICES[self.matrix]
        return kr, 1.0 - kr - kb, kb

    @property
    def spans(self) -> tuple[float, float, float]:
        """(Y' offset, Y' span, C span) in 8-bit code values."""
        if self.full_range:
            return 0.0, float(YUV_FULL_SPAN), float(YUV_FULL_SPAN)
        return float(YUV_LIMITED_Y_OFFSET), float(YUV_LIMITED_Y_SPAN), float(YUV_LIMITED_C_SPAN)


def rgb_to_yuv(rgb: npt.ArrayLike, spec: YuvSpec = YuvSpec()) -> tuple[UInt8Array, UInt8Array, UInt8Array]:
    """0-255 R'G'B' -> full-resolution 8-bit (Y', Cb, Cr) planes, rounded and clipped.

    E'Y = Kr R' + Kg G' + Kb B';  E'Cb = (B' - E'Y) / (2 (1 - Kb));  E'Cr = (R' - E'Y) / (2 (1 - Kr))
    Y' = offset + span_y E'Y;  Cb = 128 + span_c E'Cb;  Cr = 128 + span_c E'Cr
    """
    kr, kg, kb = spec.coefficients
    y_off, y_span, c_span = spec.spans
    v = np.asarray(rgb, dtype=np.float64) / _FULL_SCALE
    r, g, b = v[..., 0], v[..., 1], v[..., 2]
    ey = kr * r + kg * g + kb * b
    ecb = (b - ey) / (2.0 * (1.0 - kb))
    ecr = (r - ey) / (2.0 * (1.0 - kr))
    return (
        _to_u8(y_off + y_span * ey),
        _to_u8(YUV_C_OFFSET + c_span * ecb),
        _to_u8(YUV_C_OFFSET + c_span * ecr),
    )


def yuv_to_rgb(y: npt.ArrayLike, cb: npt.ArrayLike, cr: npt.ArrayLike, spec: YuvSpec = YuvSpec()) -> FloatArray:
    """Full-resolution (Y', Cb, Cr) planes (any dtype) -> 0-255 R'G'B' float, clipped. Exact inverse of rgb_to_yuv before rounding."""
    kr, kg, kb = spec.coefficients
    y_off, y_span, c_span = spec.spans
    ey = (np.asarray(y, dtype=np.float64) - y_off) / y_span
    ecb = (np.asarray(cb, dtype=np.float64) - YUV_C_OFFSET) / c_span
    ecr = (np.asarray(cr, dtype=np.float64) - YUV_C_OFFSET) / c_span
    r = ey + 2.0 * (1.0 - kr) * ecr
    b = ey + 2.0 * (1.0 - kb) * ecb
    g = (ey - kr * r - kb * b) / kg
    return np.clip(np.stack([r, g, b], axis=-1) * _FULL_SCALE, 0.0, _FULL_SCALE)


def subsample_chroma(plane: UInt8Array, factor: float) -> UInt8Array:
    """Downsample a chroma plane by ``factor`` per axis with an area (box) filter, 8-bit result.

    For factor 2 this is the mean of each 2x2 block: chroma sited at the
    centre of its luma block (JPEG / MPEG-1 siting).
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    h, w = plane.shape
    size = (max(1, round(w / factor)), max(1, round(h / factor)))
    return cv2.resize(np.ascontiguousarray(plane, dtype=np.uint8), size, interpolation=cv2.INTER_AREA)


def upsample_chroma(plane: npt.ArrayLike, shape: tuple[int, int], method: Upsample) -> FloatArray:
    """Upsample a chroma plane to ``shape`` (H, W).

    'nearest' replicates each chroma sample over its block; 'bilinear'
    interpolates between chroma sample centres (half-pixel aligned, matching
    the centred siting of subsample_chroma).
    """
    flags = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR}
    if method not in flags:
        raise ValueError(f"unknown upsample method {method!r}")
    src = np.asarray(plane, dtype=np.float32)
    return cv2.resize(src, (shape[1], shape[0]), interpolation=flags[method]).astype(np.float64)


def _to_u8(values: FloatArray) -> UInt8Array:
    return np.clip(np.rint(values), 0, _FULL_SCALE).astype(np.uint8)
