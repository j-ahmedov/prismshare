"""The reference frame is the experimental control: pinned, geometric twin of the codes, luminance-matched."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest

from prism_share.codec.encoder import png_bytes
from prism_share.codec.layout import base_canvas, static_mask
from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    REFERENCE_DITHER_BLOCK_PX,
    REFERENCE_TARGET_LINEAR_MEAN,
    SWEEP_CELL_PX,
    CodecParams,
)
from prism_share.colourspace import mean_linear_luminance
from prism_share.codec.decoder import read_index_band
from prism_share.codec.params import INDEX_BAND_REFERENCE
from prism_share.transmit.reference import bayer_matrix, configuration_luminance, reference_frame

FRAME_PX = CodecParams().frame_px
# Pinned: the reference must be identical for every run. A change invalidates the control.
REFERENCE_PNG_SHA256 = "322716f2e84add09c510c9a6dd289027c93de3722b994414a7b43eeedffaaddc"


def test_reference_is_pinned() -> None:
    assert hashlib.sha256(png_bytes(reference_frame(FRAME_PX))).hexdigest() == REFERENCE_PNG_SHA256


def test_reference_carries_the_reserved_index() -> None:
    """Index 0 marks the reference structurally: no capture has to be recognised by its appearance."""
    assert read_index_band(reference_frame(FRAME_PX), CodecParams()).index == INDEX_BAND_REFERENCE


def test_reference_is_deterministic_across_calls() -> None:
    reference_frame.cache_clear()
    a = np.array(reference_frame(FRAME_PX))
    reference_frame.cache_clear()
    assert np.array_equal(a, reference_frame(FRAME_PX))


def test_reference_shares_the_codes_static_region() -> None:
    mask = static_mask(FRAME_PX)
    assert np.array_equal(reference_frame(FRAME_PX)[mask], base_canvas(FRAME_PX)[mask])


def test_reference_is_pure_black_and_white() -> None:
    values = {tuple(v) for v in np.unique(reference_frame(FRAME_PX).reshape(-1, 3), axis=0)}
    assert values == {(0, 0, 0), (255, 255, 255)}


def test_reference_hits_its_linear_target() -> None:
    one_block = REFERENCE_DITHER_BLOCK_PX**2 / FRAME_PX**2
    assert abs(mean_linear_luminance(reference_frame(FRAME_PX)) - REFERENCE_TARGET_LINEAR_MEAN) <= one_block


@pytest.mark.parametrize("target", [0.15, 0.25, 0.45])
def test_reference_tracks_any_target(target: float) -> None:
    assert mean_linear_luminance(reference_frame(FRAME_PX, target)) == pytest.approx(target, abs=1e-4)


def test_reference_features_are_whole_blocks() -> None:
    """Every data-region feature is a whole 2x2 block: nothing finer than the glyphs' smallest feature."""
    ref = reference_frame(FRAME_PX)[:, :, 0]
    b = REFERENCE_DITHER_BLOCK_PX
    blocks = ref.reshape(FRAME_PX // b, b, FRAME_PX // b, b)
    assert (blocks.min(axis=(1, 3)) == blocks.max(axis=(1, 3))).all()


def test_reference_leaves_the_index_band_alone() -> None:
    """The dither may not spill into the reserved band (index 0 is all-black there)."""
    from prism_share.codec.layout import index_band_mask

    assert (reference_frame(FRAME_PX)[index_band_mask(FRAME_PX)] == 0).all()


def test_bayer_matrix() -> None:
    assert bayer_matrix(2).tolist() == [[0, 2], [3, 1]]
    m = bayer_matrix(16)
    assert sorted(m.ravel().tolist()) == list(range(256))
    with pytest.raises(ValueError):
        bayer_matrix(6)


def test_linear_not_code_value_average() -> None:
    """The brief's example: flat sRGB 128 is ~0.216 linear, a 50/50 black/white pattern is 0.5."""
    flat = np.full((4, 4, 3), 128, dtype=np.uint8)
    checker = np.zeros((4, 4, 3), dtype=np.uint8)
    checker[::2, ::2] = checker[1::2, 1::2] = 255
    assert mean_linear_luminance(flat) == pytest.approx(0.216, abs=0.001)
    assert mean_linear_luminance(checker) == pytest.approx(0.5)


def test_pinned_target_is_the_sweep_midpoint() -> None:
    """If the codec's appearance changes, the pinned target must be revisited (and this fails)."""
    means = [
        configuration_luminance(CodecParams(colour_depth=d, cell_px=c), 1.0).frame_mean
        for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX)
    ]
    midpoint = float(np.sqrt(min(means) * max(means)))
    assert REFERENCE_TARGET_LINEAR_MEAN == pytest.approx(midpoint, rel=0.005)
    worst = max(abs(np.log2(m / REFERENCE_TARGET_LINEAR_MEAN)) for m in means)
    assert worst < 0.53


def test_monochrome_is_brighter_than_colour() -> None:
    ref = mean_linear_luminance(reference_frame(FRAME_PX))
    mono = configuration_luminance(CodecParams(colour_depth=1, cell_px=8), ref)
    colour = configuration_luminance(CodecParams(colour_depth=4, cell_px=8), ref)
    assert mono.stops_vs_reference > 0 > colour.stops_vs_reference
    # Whitening makes the average content-independent: under 2 % between frames.
    assert mono.frame_spread < 0.02 * mono.frame_mean and colour.frame_spread < 0.02 * colour.frame_mean
