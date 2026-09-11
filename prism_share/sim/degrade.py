"""Synthetic channel degradations: pure functions frame -> frame.

Every degradation has the signature::

    fn(frame, severity, params, *, index=0, **options) -> frame

* ``frame``: (H, W, 3) RGB on a 0-255 scale, any dtype. Output is float64 on
  the same scale, clipped to [0, 255] wherever the physical process clips.
  Nothing is rounded except where the modelled format is itself 8-bit (the
  Y'CbCr planes); end a chain with ``quantize`` to model an 8-bit capture.
* ``severity``: one number with a physical unit (documented per function).
  Severity 0 (1 for chroma) is the identity or its closest equivalent.
* ``params``: the CodecParams of the frame. Random draws are seeded from
  ``params.seed``, the degradation's name and ``index`` (a frame number), so a
  degradation's randomness does not depend on its position in a chain.

Degradations compose in any order with ``apply_chain``. The ``Degradation``
dataclass is a declarative, hashable description of one step, used by the
pilot study and (later) the sweep.

Randomness uses numpy's PCG64 *bit stream* (guaranteed stable across numpy
versions) seeded through SeedSequence, converted to uniform / normal variates
here rather than by numpy's Generator methods, whose output may change.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from prism_share.codec.params import CodecParams
from prism_share.colourspace import (
    YuvSpec,
    linear_to_srgb,
    rgb_to_yuv,
    srgb_to_linear,
    subsample_chroma,
    upsample_chroma,
    yuv_to_rgb,
)

FloatArray = npt.NDArray[np.float64]
_FULL_SCALE = 255.0

# --------------------------------------------------------------------------- #
# Seeded variates
# --------------------------------------------------------------------------- #


def _bit_generator(params: CodecParams, name: str, index: int) -> np.random.PCG64:
    entropy = [params.seed, zlib.crc32(name.encode("ascii")), index]
    return np.random.PCG64(np.random.SeedSequence(entropy))


def _uniform(params: CodecParams, name: str, index: int, n: int) -> FloatArray:
    """``n`` uniforms in (0, 1) from the 53 high bits of each raw 64-bit draw."""
    raw = _bit_generator(params, name, index).random_raw(n)
    return ((raw >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def _normal(params: CodecParams, name: str, index: int, shape: tuple[int, ...]) -> FloatArray:
    """Standard normal variates by Box-Muller."""
    n = int(np.prod(shape))
    half = -(-n // 2)
    u = _uniform(params, name, index, 2 * half)
    radius = np.sqrt(-2.0 * np.log(u[:half]))
    angle = 2.0 * np.pi * u[half:]
    z = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
    return z[:n].reshape(shape)


def _as_float(frame: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {arr.shape}")
    return arr


# --------------------------------------------------------------------------- #
# Degradations
# --------------------------------------------------------------------------- #


def gaussian_blur(frame: npt.ArrayLike, severity: float, params: CodecParams, *, index: int = 0) -> FloatArray:
    """Optical defocus / lens PSF. ``severity`` = Gaussian sigma in screen pixels."""
    img = _as_float(frame)
    if severity <= 0:
        return img.copy()
    out = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=severity, sigmaY=severity, borderType=cv2.BORDER_REPLICATE)
    return out.astype(np.float64)


def sensor_noise(frame: npt.ArrayLike, severity: float, params: CodecParams, *, index: int = 0) -> FloatArray:
    """Additive white Gaussian noise, independent per pixel and channel, then clipping.

    ``severity`` = noise standard deviation in 8-bit code values.
    """
    img = _as_float(frame)
    if severity <= 0:
        return img.copy()
    return np.clip(img + severity * _normal(params, "noise", index, img.shape), 0.0, _FULL_SCALE)


def perspective_warp(frame: npt.ArrayLike, severity: float, params: CodecParams, *, index: int = 0) -> FloatArray:
    """Oblique view followed by ideal rectification: the resampling loss of a tilted camera.

    Each frame corner moves inwards along x and y by an independent uniform
    fraction of ``severity * frame_px`` (so ``severity`` = maximum corner
    displacement as a fraction of the frame side). The frame is warped onto
    that quadrilateral (bilinear, black surround) at unchanged resolution, then
    warped back with the exact inverse homography (bilinear). This isolates
    foreshortening + double interpolation; it does not model detection error,
    which detect.py will contribute for real captures.
    """
    img = _as_float(frame)
    if severity <= 0:
        return img.copy()
    h, w = img.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
    shift = _uniform(params, "perspective", index, src.size).reshape(src.shape) * severity * min(h, w)
    dst = src + inward * shift
    homography = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    img32 = img.astype(np.float32)
    warped = cv2.warpPerspective(img32, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    back = cv2.warpPerspective(
        warped, homography, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )
    return np.clip(back.astype(np.float64), 0.0, _FULL_SCALE)


def white_balance_shift(frame: npt.ArrayLike, severity: float, params: CodecParams, *, index: int = 0) -> FloatArray:
    """Auto-white-balance error: red/blue gain imbalance applied in linear light.

    ``severity`` = imbalance in stops: red gain 2^(s/2), blue gain 2^(-s/2),
    green 1 (warm shift; negative severity = cool shift). Values are decoded
    with the sRGB EOTF, scaled, clipped at sensor full scale, re-encoded.
    """
    img = _as_float(frame)
    if severity == 0:
        return img.copy()
    gains = np.array([2.0 ** (severity / 2), 1.0, 2.0 ** (-severity / 2)])
    linear = np.clip(srgb_to_linear(img / _FULL_SCALE) * gains, 0.0, 1.0)
    return linear_to_srgb(linear) * _FULL_SCALE


def chroma_subsample(
    frame: npt.ArrayLike,
    severity: float,
    params: CodecParams,
    *,
    index: int = 0,
    matrix: str = YuvSpec().matrix,
    full_range: bool = False,
    upsample: str = "nearest",
) -> FloatArray:
    """YUV_420_888 round trip: RGB -> 8-bit Y'CbCr, chroma subsampled, upsampled, -> RGB.

    ``severity`` = chroma sample pitch in screen pixels. 1 = 4:4:4 (only the
    8-bit Y'CbCr quantisation), 2 = 4:2:0 with the camera sampling the screen
    1:1 (YUV_420_FACTOR). A camera with m sensor pixels per screen pixel has an
    effective pitch of 2/m, so values below 2 model magnified captures.
    ``upsample`` = 'nearest' or 'bilinear': how the consumer reconstructs chroma.
    """
    img = _as_float(frame)
    if severity < 1:
        raise ValueError("chroma sample pitch must be >= 1 screen pixel")
    spec = YuvSpec(matrix=matrix, full_range=full_range)
    y, cb, cr = rgb_to_yuv(img, spec)
    if severity > 1:
        cb = upsample_chroma(subsample_chroma(cb, severity), y.shape, upsample)  # type: ignore[arg-type]
        cr = upsample_chroma(subsample_chroma(cr, severity), y.shape, upsample)  # type: ignore[arg-type]
    return yuv_to_rgb(y, cb, cr, spec)


def quantize(frame: npt.ArrayLike, severity: float = 0.0, params: CodecParams | None = None, *, index: int = 0) -> FloatArray:
    """Round to integers and clip to [0, 255]: an 8-bit capture. ``severity`` is ignored."""
    return np.clip(np.rint(_as_float(frame)), 0.0, _FULL_SCALE)


DEGRADATIONS: dict[str, Callable[..., FloatArray]] = {
    "blur": gaussian_blur,
    "noise": sensor_noise,
    "perspective": perspective_warp,
    "white_balance": white_balance_shift,
    "chroma": chroma_subsample,
    "quantize": quantize,
}

#: Physical unit of each degradation's severity, for tables and plot axes.
SEVERITY_UNITS: dict[str, str] = {
    "blur": "Gaussian sigma (screen px)",
    "noise": "noise sigma (8-bit code values)",
    "perspective": "max corner displacement (fraction of frame side)",
    "white_balance": "R/B gain imbalance (stops)",
    "chroma": "chroma sample pitch (screen px); 2 = 4:2:0 at 1:1",
    "quantize": "(none)",
}


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Degradation:
    """One step of a degradation chain: name, severity and keyword options."""

    name: str
    severity: float
    options: tuple[tuple[str, Any], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.name not in DEGRADATIONS:
            raise ValueError(f"unknown degradation {self.name!r}; choose from {sorted(DEGRADATIONS)}")

    @classmethod
    def of(cls, name: str, severity: float, **options: Any) -> Degradation:
        return cls(name, float(severity), tuple(sorted(options.items())))

    def __call__(self, frame: npt.ArrayLike, params: CodecParams, *, index: int = 0) -> FloatArray:
        return DEGRADATIONS[self.name](frame, self.severity, params, index=index, **dict(self.options))

    @property
    def label(self) -> str:
        opts = ",".join(f"{k}={v}" for k, v in self.options)
        return f"{self.name}({self.severity:g}{',' + opts if opts else ''})"


def apply_chain(
    frame: npt.ArrayLike, chain: Sequence[Degradation], params: CodecParams, *, index: int = 0
) -> FloatArray:
    """Apply degradations left to right."""
    out = _as_float(frame)
    for step in chain:
        out = step(out, params, index=index)
    return out
