"""The index band: every frame says which frame it is, independently of how well it decodes."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from prism_share.codec.decoder import read_index_band, read_symbols
from prism_share.codec.encoder import INDEX_BAND_CODE_INDICES, band_index, encode_frames, render_frame
from prism_share.codec.layout import draw_index_band, index_band, index_band_mask
from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    INDEX_BAND_BITS,
    INDEX_BAND_REFERENCE,
    INDEX_BAND_REPEATS,
    SWEEP_CELL_PX,
    CodecParams,
)

ALL_CONFIGS = [CodecParams(colour_depth=d, cell_px=c) for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX)]


def _frame_with_index(params: CodecParams, index: int) -> np.ndarray:
    truth = encode_frames(b"index", params, n_frames=1)[0]
    return render_frame(truth.glyphs, truth.colours, params, index)


@pytest.mark.parametrize("index", [0, 1, 2, 127, 128, 200, 255])
def test_every_index_round_trips(index: int) -> None:
    params = CodecParams(colour_depth=16, cell_px=4)
    assert read_index_band(_frame_with_index(params, index), params).index == index


@pytest.mark.parametrize("params", ALL_CONFIGS, ids=lambda p: p.label)
def test_readable_at_every_configuration(params: CodecParams) -> None:
    for index in (1, 170, 255):
        readout = read_index_band(_frame_with_index(params, index), params)
        assert readout.index == index and readout.agreement == 1.0


def test_encoded_frames_carry_their_block_id() -> None:
    params = CodecParams()
    for frame in encode_frames(b"blocks", params, n_frames=5):
        assert frame.frame_index == frame.header.block_id + 1
        assert read_index_band(frame.image, params).index == frame.frame_index


def test_index_wraps_rather_than_failing() -> None:
    """A real payload can need thousands of frames; the band is 8 bits."""
    assert band_index(0) == 1 and band_index(INDEX_BAND_CODE_INDICES - 1) == 2**INDEX_BAND_BITS - 1
    assert band_index(INDEX_BAND_CODE_INDICES) == 1  # wrapped
    assert INDEX_BAND_REFERENCE not in {band_index(i) for i in range(1000)}
    with pytest.raises(ValueError):
        band_index(-1)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0])
def test_survives_blur_that_destroys_the_data(sigma: float) -> None:
    """The band must read in exactly the cases that matter: badly degraded captures."""
    params = CodecParams(colour_depth=16, cell_px=4)
    truth = encode_frames(b"degraded", params, n_frames=1)[0]
    blurred = cv2.GaussianBlur(truth.image.astype(np.float64), (0, 0), sigma)
    assert np.mean(read_symbols(blurred, params).glyphs != truth.glyphs) > 0.2  # data unreadable
    assert read_index_band(blurred, params).index == truth.frame_index  # index still exact


def test_survives_noise_and_a_dim_capture() -> None:
    params = CodecParams(colour_depth=4, cell_px=5)
    truth = encode_frames(b"noisy", params, n_frames=1)[0]
    rng = np.random.default_rng(0)
    degraded = np.clip(truth.image * 0.35 + 12 + rng.normal(0, 25, truth.image.shape), 0, 255)
    assert read_index_band(degraded, params).index == truth.frame_index


def test_majority_vote_survives_one_corrupted_copy() -> None:
    """One of the three copies is painted over; the other two still decide."""
    params = CodecParams()
    truth = encode_frames(b"vote", params, n_frames=1)[0]
    frame = truth.image.copy()
    band = index_band(params.frame_px)
    per_copy = INDEX_BAND_BITS * band.block_px
    frame[band.y : band.y + band.height, band.x : band.x + per_copy] = 255  # first copy: all ones
    readout = read_index_band(frame, params)
    assert readout.index == truth.frame_index
    assert readout.agreement == pytest.approx((INDEX_BAND_REPEATS - 1) / INDEX_BAND_REPEATS, abs=0.2)


def test_two_corrupted_copies_are_reported_by_agreement() -> None:
    params = CodecParams()
    truth = encode_frames(b"vote", params, n_frames=1)[0]
    frame = truth.image.copy()
    band = index_band(params.frame_px)
    frame[band.y : band.y + band.height, band.x : band.x + 2 * INDEX_BAND_BITS * band.block_px] = 255
    readout = read_index_band(frame, params)
    assert readout.index == 2**INDEX_BAND_BITS - 1  # the two bad copies win, as majority voting must
    assert readout.agreement < 1.0  # and the disagreement is reported


def test_reading_at_rectification_scale_k() -> None:
    params = CodecParams(colour_depth=8, cell_px=6)
    truth = encode_frames(b"scale", params, n_frames=1)[0]
    for k in (1, 2, 3):
        upscaled = np.kron(truth.image, np.ones((k, k, 1)))
        assert read_index_band(upscaled, params).index == truth.frame_index


def test_band_is_not_confused_by_cell_content() -> None:
    """Whatever the cells say, the band says the index: two frames of the same
    configuration differ in the band only by their index."""
    params = CodecParams(colour_depth=16, cell_px=4)
    a, b = encode_frames(b"different payload entirely", params, n_frames=2)
    mask = index_band_mask(params.frame_px)
    expected = np.array(a.image)
    draw_index_band(expected, b.frame_index, params.frame_px)
    assert np.array_equal(b.image[mask], expected[mask])
