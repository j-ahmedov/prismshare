"""Payload bytes -> frame images, and a deterministic PNG writer.

Pipeline: payload -> fountain source blocks -> one encoded block per frame ->
framing (header, CRC, RS, interleave, whitening) -> cell symbols -> pixels.

Frames are (frame_px, frame_px, 3) uint8 **RGB** arrays. Convert to BGR only
at an OpenCV I/O boundary.

PNGs are written by ``png_bytes``, a minimal encoder using *stored*
(uncompressed) deflate blocks. Compressed PNG output depends on the zlib build
(zlib vs zlib-ng produce different bytes for the same pixels); stored blocks,
CRC-32 and Adler-32 are fully specified, so the file bytes are a pure function
of the pixels on every machine. The cost is size: ~3 MB per 1024 px frame.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from prism_share.codec.fountain import (
    default_frame_count,
    encode_block,
    n_source_blocks,
    split_payload,
)
from prism_share.codec.framing import FrameHeader, encode_frame_symbols, frame_capacity
from prism_share.codec.glyphs import glyphs_for
from prism_share.codec.layout import base_canvas, cell_pixel_index
from prism_share.codec.palette import palette_for
from prism_share.codec.params import BACKGROUND_RGB, CodecParams

UInt8Array = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class EncodedFrame:
    """One frame plus its ground truth."""

    header: FrameHeader
    glyphs: IntArray
    """(n_cells,) glyph index per cell."""
    colours: IntArray
    """(n_cells,) colour index per cell."""
    image: UInt8Array
    """(frame_px, frame_px, 3) uint8 RGB."""


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def cell_tiles(params: CodecParams) -> UInt8Array:
    """Every (glyph, colour) cell bitmap: (glyph_count, colour_depth, cell_px, cell_px, 3) RGB."""
    glyphs = glyphs_for(params)
    ink = palette_for(params)
    background = np.array(BACKGROUND_RGB, dtype=np.uint8)
    tiles = np.where(
        glyphs[:, None, :, :, None],  # (G, 1, N, N, 1)
        ink[None, :, None, None, :],  # (1, C, 1, 1, 3)
        background,
    )
    return tiles.astype(np.uint8)


def render_frame(glyphs: IntArray, colours: IntArray, params: CodecParams) -> UInt8Array:
    """Draw a frame from per-cell symbol indices. Pure function of its arguments."""
    glyphs = np.asarray(glyphs, dtype=np.int64)
    colours = np.asarray(colours, dtype=np.int64)
    rows, cols = cell_pixel_index(params)
    if glyphs.shape != (len(rows),) or colours.shape != (len(rows),):
        raise ValueError(f"expected {len(rows)} symbols per frame")
    if glyphs.min() < 0 or glyphs.max() >= params.glyph_count:
        raise ValueError("glyph index out of range")
    if colours.min() < 0 or colours.max() >= params.colour_depth:
        raise ValueError("colour index out of range")
    frame = np.array(base_canvas(params.frame_px))  # writable copy
    frame[rows, cols] = cell_tiles(params)[glyphs, colours]
    return frame


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def encode_frames(payload: bytes, params: CodecParams, *, n_frames: int | None = None) -> list[EncodedFrame]:
    """Encode ``payload`` into frames with ground truth.

    Frames 0..K-1 carry the source blocks; later frames are fountain repair
    blocks. ``n_frames`` defaults to ``fountain.default_frame_count(K)``.
    """
    cap = frame_capacity(params)
    source = split_payload(payload, cap.block_bytes)
    k = n_source_blocks(len(payload), cap.block_bytes)
    total = default_frame_count(k) if n_frames is None else n_frames
    if total < 1:
        raise ValueError("n_frames must be >= 1")
    fingerprint = params.fingerprint()
    frames: list[EncodedFrame] = []
    for block_id in range(total):
        header = FrameHeader(block_id, k, len(payload), fingerprint)
        glyphs, colours = encode_frame_symbols(header, encode_block(source, block_id, params.seed), params)
        frames.append(EncodedFrame(header, glyphs, colours, render_frame(glyphs, colours, params)))
    return frames


def encode(payload: bytes, params: CodecParams, *, n_frames: int | None = None) -> list[UInt8Array]:
    """Encode ``payload`` into a list of (frame_px, frame_px, 3) uint8 RGB frames."""
    return [f.image for f in encode_frames(payload, params, n_frames=n_frames)]


# --------------------------------------------------------------------------- #
# Deterministic PNG
# --------------------------------------------------------------------------- #

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ZLIB_HEADER = b"\x78\x01"  # deflate, 32 KiB window, no dictionary, fastest
_STORED_BLOCK_MAX = 0xFFFF
_PNG_COLOUR_TYPE_RGB = 2
_PNG_FILTER_NONE = 0
_U32 = 0xFFFFFFFF


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & _U32)


def _stored_zlib(raw: bytes) -> bytes:
    out = bytearray(_ZLIB_HEADER)
    for start in range(0, len(raw), _STORED_BLOCK_MAX):
        chunk = raw[start : start + _STORED_BLOCK_MAX]
        final = start + _STORED_BLOCK_MAX >= len(raw)
        out += struct.pack("<BHH", int(final), len(chunk), len(chunk) ^ 0xFFFF) + chunk
    out += struct.pack(">I", zlib.adler32(raw) & _U32)
    return bytes(out)


def png_bytes(image: UInt8Array) -> bytes:
    """Encode an (H, W, 3) uint8 RGB image as PNG with byte-exact reproducibility."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected (H, W, 3) uint8 RGB")
    height, width, _ = image.shape
    bit_depth = np.iinfo(np.uint8).bits
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, _PNG_COLOUR_TYPE_RGB, 0, 0, 0)
    rows = np.empty((height, 1 + width * image.shape[2]), dtype=np.uint8)
    rows[:, 0] = _PNG_FILTER_NONE
    rows[:, 1:] = image.reshape(height, -1)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", _stored_zlib(rows.tobytes()))
        + _png_chunk(b"IEND", b"")
    )


def write_png(path: str | Path, image: UInt8Array) -> None:
    Path(path).write_bytes(png_bytes(image))


def save_frames(frames: list[UInt8Array], directory: str | Path) -> list[Path]:
    """Write frames as ``frame_00000.png``, ... into ``directory``; returns the paths."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, image in enumerate(frames):
        path = out_dir / f"frame_{i:05d}.png"
        write_png(path, image)
        paths.append(path)
    return paths
