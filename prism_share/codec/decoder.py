"""Rectified frame -> symbol stream with per-symbol confidence.

Pure functions: no I/O, no camera, no global mutable state. The input is a
frame already rectified to (frame_px, frame_px) - by detect.py for captures,
or the encoder's own output for synthetic tests. Pixel values are on a 0-255
scale, either RGB (H, W, 3) or a single intensity plane (H, W) such as the Y
plane of a YUV capture. A single plane can only be decoded for monochrome codes.

Algorithm
---------
1. **Level normalisation** (``normalise=True``, the default). Per channel,
   ``(x - black) / (white - black)`` clipped to [0, 1], with ``white`` the
   median of the white border band and ``black`` the median of the fiducials'
   black border modules (layout.white_reference_mask / black_reference_mask).
   Both references lie in the static region, so the correction is computed
   identically for every configuration. Per-channel scaling to the reference
   white is also a von Kries white balance against the display's own white.
2. **Shape.** Cell intensity is the per-pixel maximum over channels (HSV
   value). Every palette colour has a full-scale channel, so every ink colour
   has the same contrast against the black background under this measure. The
   cell is correlated (Pearson) with each glyph bitmap; the glyph is the
   argmax, and **glyph confidence** = best - second-best correlation, in [0, 2].
3. **Colour**, independently of the shape decision: the ``glyph_weight``
   brightest pixels of the cell are taken as ink (every glyph has exactly that
   many ink pixels); their mean RGB is scaled so its largest channel is 1, and
   the nearest palette colour (also scaled, Euclidean) is chosen. **Colour
   confidence** = second-nearest - nearest distance. Because colour never uses
   the decoded glyph, a shape error cannot cause a colour error or vice versa
   through the decoder itself - the two error rates measure different physics.
   For colour_depth 1 no colour decision is made: colour 0, confidence +inf.

Variants (``DecoderOptions``), compared in the pilot study (docs/pilot.md):

* ``shape_channel='luma'`` reads shape (and ranks ink pixels) from Y' instead
  of the channel maximum. Immune to chroma subsampling, weaker for dark-luma
  inks (pure blue) under noise and blur.
* ``colour_estimator='saturated'`` averages only the most saturated
  ``COLOUR_CORE_FRACTION`` of the ink pixels: the ones least diluted by
  neighbouring black under 4:2:0. Much better under chroma subsampling, much
  worse under strong noise.

The default (``max``/``mean``) is the algorithm described above. No variant is
best under every degradation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prism_share.codec.fountain import decode_blocks
from prism_share.codec.framing import FrameDecodeResult, decode_frame_symbols, frame_capacity
from prism_share.codec.glyphs import glyphs_for
from prism_share.codec.layout import black_reference_mask, cell_pixel_index, white_reference_mask
from prism_share.codec.palette import palette_for
from prism_share.codec.params import COLOUR_CORE_FRACTION, DECODER_LUMA_MATRIX, YUV_MATRICES, CodecParams

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

_FULL_SCALE = float(np.iinfo(np.uint8).max)
_EPS = 1e-9


@dataclass(frozen=True)
class SymbolReadout:
    """Per-cell decisions for one frame, in cell-index order."""

    glyphs: IntArray
    colours: IntArray
    glyph_confidence: FloatArray
    """Best minus second-best glyph correlation, in [0, 2]. 0 = coin toss."""
    colour_confidence: FloatArray
    """Second-nearest minus nearest palette distance, >= 0; +inf when colour_depth == 1."""


@dataclass(frozen=True)
class DecoderOptions:
    """Decoder variants. These are analysis choices, not code parameters."""

    normalise: bool = True
    """Map reference black/white to 0/1 per channel (see module docstring)."""
    shape_channel: str = "max"
    """'max' = per-pixel channel maximum; 'luma' = Y' of the normalised RGB
    (DECODER_LUMA_MATRIX). Also the channel that ranks pixels as ink."""
    colour_estimator: str = "mean"
    """'mean' = mean of all ink pixels; 'saturated' = mean of the most
    saturated COLOUR_CORE_FRACTION of them."""

    def __post_init__(self) -> None:
        if self.shape_channel not in ("max", "luma"):
            raise ValueError(f"shape_channel must be 'max' or 'luma', got {self.shape_channel!r}")
        if self.colour_estimator not in ("mean", "saturated"):
            raise ValueError(f"colour_estimator must be 'mean' or 'saturated', got {self.colour_estimator!r}")

    @property
    def label(self) -> str:
        return f"{self.shape_channel}/{self.colour_estimator}" + ("" if self.normalise else "/raw")


def normalise_levels(frame: FloatArray, frame_px: int) -> FloatArray:
    """Map the reference black to 0 and reference white to 1, per channel, clipped."""
    white = np.median(frame[white_reference_mask(frame_px)], axis=0)
    black = np.median(frame[black_reference_mask(frame_px)], axis=0)
    span = np.maximum(white - black, _EPS)
    return np.clip((frame - black) / span, 0.0, 1.0)


def read_symbols(
    rectified: npt.ArrayLike, params: CodecParams, options: DecoderOptions = DecoderOptions()
) -> SymbolReadout:
    """Read every cell of a rectified frame."""
    frame = np.asarray(rectified, dtype=np.float64)
    if frame.shape[:2] != (params.frame_px, params.frame_px) or frame.ndim not in (2, 3):
        raise ValueError(f"expected ({params.frame_px}, {params.frame_px}[, 3]), got {frame.shape}")
    is_rgb = frame.ndim == 3
    if is_rgb and frame.shape[2] != 3:
        raise ValueError("colour input must have 3 channels (RGB)")
    if not is_rgb and params.colour_depth > 1:
        raise ValueError("a single intensity plane cannot be decoded for colour_depth > 1")
    frame = normalise_levels(frame, params.frame_px) if options.normalise else frame / _FULL_SCALE

    rows, cols = cell_pixel_index(params)
    n_cells = len(rows)
    cell_pixels = params.cell_px * params.cell_px
    patches = frame[rows, cols].reshape(n_cells, cell_pixels, -1)  # (n, N*N, channels)
    if not is_rgb or options.shape_channel == "max":
        intensity = patches.max(axis=2)  # (n, N*N)
    else:
        kr, kb = YUV_MATRICES[DECODER_LUMA_MATRIX]
        intensity = patches @ np.array([kr, 1.0 - kr - kb, kb])

    glyphs, glyph_conf = _classify_shape(intensity, params)
    if params.colour_depth == 1:
        colours = np.zeros(n_cells, dtype=np.int64)
        colour_conf = np.full(n_cells, np.inf)
    else:
        colours, colour_conf = _classify_colour(patches, intensity, params, options.colour_estimator)
    return SymbolReadout(glyphs, colours, glyph_conf, colour_conf)


def _classify_shape(intensity: FloatArray, params: CodecParams) -> tuple[IntArray, FloatArray]:
    templates = np.where(glyphs_for(params).reshape(params.glyph_count, -1), 1.0, -1.0)
    templates -= templates.mean(axis=1, keepdims=True)
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)

    centred = intensity - intensity.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    safe = np.where(norms > _EPS, norms, 1.0)
    scores = (centred @ templates.T) / safe[:, None]
    scores[norms <= _EPS] = 0.0  # flat cell: no shape information at all

    order = np.argsort(-scores, axis=1, kind="stable")
    best = np.take_along_axis(scores, order[:, :1], axis=1)[:, 0]
    second = np.take_along_axis(scores, order[:, 1:2], axis=1)[:, 0]
    return order[:, 0].astype(np.int64), best - second


def _classify_colour(
    patches: FloatArray, intensity: FloatArray, params: CodecParams, estimator: str
) -> tuple[IntArray, FloatArray]:
    ink_idx = np.argsort(-intensity, axis=1, kind="stable")[:, : params.glyph_weight]
    ink = np.take_along_axis(patches, ink_idx[:, :, None], axis=1)  # (n, weight, 3)
    if estimator == "saturated":
        core = max(1, round(params.glyph_weight * COLOUR_CORE_FRACTION))
        saturation = ink.max(axis=2) - ink.min(axis=2)
        core_idx = np.argsort(-saturation, axis=1, kind="stable")[:, :core]
        ink = np.take_along_axis(ink, core_idx[:, :, None], axis=1)
    ink_rgb = ink.mean(axis=1)  # (n, 3)
    peak = ink_rgb.max(axis=1, keepdims=True)
    chroma = ink_rgb / np.maximum(peak, _EPS)

    ref = palette_for(params).astype(np.float64)
    ref /= ref.max(axis=1, keepdims=True)
    dist = np.linalg.norm(chroma[:, None, :] - ref[None, :, :], axis=2)  # (n, D)

    order = np.argsort(dist, axis=1, kind="stable")
    nearest = np.take_along_axis(dist, order[:, :1], axis=1)[:, 0]
    runner_up = np.take_along_axis(dist, order[:, 1:2], axis=1)[:, 0]
    return order[:, 0].astype(np.int64), runner_up - nearest


def decode_frame(
    rectified: npt.ArrayLike, params: CodecParams, options: DecoderOptions = DecoderOptions()
) -> FrameDecodeResult:
    """Read symbols and decode the frame's RS codewords, header and CRC."""
    readout = read_symbols(rectified, params, options)
    return decode_frame_symbols(readout.glyphs, readout.colours, params)


def decode_payload(rectified_frames: Iterable[npt.ArrayLike], params: CodecParams) -> bytes | None:
    """Recover the payload from any sufficient set of rectified frames, or None.

    Frames that fail to decode are skipped. The header of the first good frame
    fixes the source-block count and payload length; frames disagreeing with it
    are ignored.
    """
    block_bytes = frame_capacity(params).block_bytes
    received: dict[int, bytes] = {}
    n_source = payload_len = None
    for rectified in rectified_frames:
        result = decode_frame(rectified, params)
        if not result.ok or result.header is None or result.block is None:
            continue
        if n_source is None:
            n_source, payload_len = result.header.n_source_blocks, result.header.payload_len
        if (result.header.n_source_blocks, result.header.payload_len) != (n_source, payload_len):
            continue
        received[result.header.block_id] = result.block
    if n_source is None or payload_len is None:
        return None
    blocks = decode_blocks(received, n_source, block_bytes, params.seed)
    if blocks is None:
        return None
    return blocks.tobytes()[:payload_len]
