from __future__ import annotations

import numpy as np
import pytest

from prism_share.codec import glyphs
from prism_share.codec.params import GLYPH_ALGORITHM_VERSION, SWEEP_CELL_PX, CodecParams


@pytest.mark.parametrize("cell_px", SWEEP_CELL_PX)
def test_glyph_set_properties(cell_px: int) -> None:
    p = CodecParams(cell_px=cell_px)
    gs = glyphs.glyph_set(p.glyph_count, cell_px)
    pats = gs.patterns
    assert pats.shape == (p.glyph_count, cell_px, cell_px)
    assert pats.dtype == bool
    # Constant weight: every glyph has exactly glyph_weight ink pixels.
    assert (pats.reshape(len(pats), -1).sum(axis=1) == p.glyph_weight).all()
    # All distinct, and the reported minimum distance is the true one.
    assert len({np.packbits(g).tobytes() for g in pats}) == p.glyph_count
    assert glyphs.min_pairwise_hamming(pats) == (gs.min_hamming, gs.pairs_at_min)
    # No single-pixel features.
    assert not any(glyphs._has_isolated_pixel(g) for g in pats)
    # Canonical order.
    strings = [glyphs._bitstring(g) for g in pats]
    assert strings == sorted(strings)


@pytest.mark.parametrize("cell_px", SWEEP_CELL_PX)
def test_minimum_distance_is_useful(cell_px: int) -> None:
    """At least 30 % of the cell's pixels must differ between any two glyphs."""
    gs = glyphs.glyph_set(16, cell_px)
    assert gs.min_hamming >= 0.3 * cell_px * cell_px


def test_cell_px_4_reaches_the_plotkin_bound() -> None:
    # Plotkin: 16 binary words of length 16 have d <= 16*16 / (2*15) = 8.53 -> 8.
    assert glyphs.glyph_set(16, 4).min_hamming == 8


@pytest.mark.parametrize("cell_px", SWEEP_CELL_PX)
def test_committed_cache_matches_regeneration(cell_px: int) -> None:
    """Guards the thesis glyph sets: the cache must be exactly what the algorithm produces."""
    path = glyphs.cache_path(16, cell_px)
    assert path.exists(), f"glyph cache missing: {path}"
    cached = glyphs._read_cache(path)
    assert cached.algorithm_version == GLYPH_ALGORITHM_VERSION
    fresh = glyphs.generate_glyph_set(16, cell_px)
    assert np.array_equal(cached.patterns, fresh.patterns)
    assert (cached.min_hamming, cached.pairs_at_min, cached.pool_size) == (
        fresh.min_hamming,
        fresh.pairs_at_min,
        fresh.pool_size,
    )


def test_glyphs_for_uses_params() -> None:
    assert glyphs.glyphs_for(CodecParams(cell_px=6)).shape == (16, 6, 6)


def test_glyph_set_is_independent_of_payload_seed() -> None:
    a = glyphs.glyphs_for(CodecParams(seed=0))
    b = glyphs.glyphs_for(CodecParams(seed=12345))
    assert np.array_equal(a, b)


def test_glyphs_are_read_only() -> None:
    with pytest.raises(ValueError):
        glyphs.glyphs_for(CodecParams())[0, 0, 0] = True
