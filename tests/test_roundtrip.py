"""Build step 2 gate: synthetic encode -> decode, no camera, every colour depth x cell size."""

from __future__ import annotations

import itertools
import random

import cv2
import numpy as np
import pytest

from prism_share.codec.decoder import decode_frame, decode_payload, read_symbols
from prism_share.codec.encoder import encode, encode_frames
from prism_share.codec.framing import frame_capacity
from prism_share.codec.params import ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX, CodecParams

GRID = [CodecParams(colour_depth=d, cell_px=c) for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX)]


@pytest.mark.parametrize("params", GRID, ids=lambda p: p.label)
def test_symbols_round_trip_exactly(params: CodecParams) -> None:
    payload = random.Random(params.label).randbytes(frame_capacity(params).block_bytes * 2)
    for frame in encode_frames(payload, params, n_frames=3):
        readout = read_symbols(frame.image, params)
        assert np.array_equal(readout.glyphs, frame.glyphs)
        assert np.array_equal(readout.colours, frame.colours)
        # A clean frame is maximally confident: every cell correlates perfectly with
        # its own glyph, and colour lands exactly on a palette entry.
        assert np.all(readout.glyph_confidence > 0.5)
        assert np.all(readout.colour_confidence > 0.4)
        result = decode_frame(frame.image, params)
        assert result.ok and result.header == frame.header and result.codewords_failed == 0


@pytest.mark.parametrize("params", GRID, ids=lambda p: p.label)
def test_payload_round_trip(params: CodecParams) -> None:
    block = frame_capacity(params).block_bytes
    payload = random.Random(params.label).randbytes(int(block * 2.5))
    frames = encode(payload, params)
    assert decode_payload(frames, params) == payload


@pytest.mark.parametrize("params", [GRID[0], GRID[-1]], ids=lambda p: p.label)
def test_payload_survives_lost_frames(params: CodecParams) -> None:
    block = frame_capacity(params).block_bytes
    payload = random.Random(1).randbytes(block * 6 - 11)
    frames = encode(payload, params, n_frames=12)
    survivors = [f for i, f in enumerate(frames) if i not in {0, 2, 3, 5}]  # 4 of 6 source frames lost
    assert decode_payload(survivors, params) == payload
    assert decode_payload(list(reversed(survivors)), params) == payload


def test_too_few_frames_returns_none() -> None:
    p = CodecParams()
    frames = encode(random.Random(2).randbytes(frame_capacity(p).block_bytes * 3), p)
    assert decode_payload(frames[:2], p) is None


def test_empty_payload() -> None:
    p = CodecParams(colour_depth=1)
    assert decode_payload(encode(b"", p), p) == b""


@pytest.mark.parametrize("cell_px", SWEEP_CELL_PX)
def test_monochrome_decodes_from_single_plane(cell_px: int) -> None:
    """The mechanism experiment feeds the Y plane of YUV captures directly."""
    p = CodecParams(colour_depth=1, cell_px=cell_px)
    frame = encode(b"luma", p, n_frames=1)[0]
    y_plane = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    assert decode_frame(y_plane, p).ok


def test_single_plane_rejected_for_colour() -> None:
    p = CodecParams(colour_depth=4)
    frame = encode(b"x", p, n_frames=1)[0]
    with pytest.raises(ValueError):
        read_symbols(frame[:, :, 0], p)


def test_decoder_accepts_float_input_and_is_level_invariant() -> None:
    """A dim, offset capture of the same frame decodes identically (normalisation)."""
    p = CodecParams(colour_depth=16, cell_px=5)
    frame = encode(b"levels", p, n_frames=1)[0]
    dim = frame.astype(np.float64) * 0.6 + 20.0
    ref = read_symbols(frame, p)
    got = read_symbols(dim, p)
    assert np.array_equal(ref.glyphs, got.glyphs) and np.array_equal(ref.colours, got.colours)


def test_decoder_rejects_wrong_shape() -> None:
    p = CodecParams()
    with pytest.raises(ValueError):
        read_symbols(np.zeros((p.frame_px, p.frame_px - 1, 3), dtype=np.uint8), p)


def test_frame_decoded_with_wrong_params_fails() -> None:
    p = CodecParams()
    frame = encode(b"x", p, n_frames=1)[0]
    assert not decode_frame(frame, p.replace(seed=1)).ok
