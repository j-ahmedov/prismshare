from __future__ import annotations

import itertools

import numpy as np
import pytest

from prism_share.codec import palette as pal
from prism_share.codec.params import ALLOWED_COLOUR_DEPTHS, MONO_INK_RGB, CodecParams

# Pinned so a change to the selection rule cannot silently change the thesis palettes.
EXPECTED = {
    1: [(255, 255, 255)],
    2: [(0, 255, 0), (255, 0, 255)],
    4: [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)],
    8: [
        (0, 255, 0), (0, 255, 255), (128, 128, 255), (128, 255, 128),
        (255, 0, 255), (255, 128, 128), (255, 255, 0), (255, 255, 255),
    ],
    16: [
        (0, 128, 255), (0, 255, 0), (0, 255, 128), (0, 255, 255),
        (128, 128, 255), (128, 255, 0), (128, 255, 128), (128, 255, 255),
        (255, 0, 128), (255, 0, 255), (255, 128, 0), (255, 128, 128),
        (255, 128, 255), (255, 255, 0), (255, 255, 128), (255, 255, 255),
    ],
}  # fmt: skip


@pytest.mark.parametrize("depth", ALLOWED_COLOUR_DEPTHS)
def test_palette_is_pinned(depth: int) -> None:
    got = [tuple(int(v) for v in c) for c in pal.palette(depth)]
    assert got == EXPECTED[depth]


@pytest.mark.parametrize("depth", ALLOWED_COLOUR_DEPTHS)
def test_palette_invariants(depth: int) -> None:
    p = pal.palette_for(CodecParams(colour_depth=depth))
    assert p.shape == (depth, 3) and p.dtype == np.uint8
    assert len({tuple(c) for c in p}) == depth
    # Every ink colour has a full-scale channel: equal shape contrast on black.
    assert (p.max(axis=1) == 255).all()


def test_monochrome_is_white() -> None:
    assert tuple(pal.palette(1)[0]) == MONO_INK_RGB


@pytest.mark.parametrize("depth", [2, 4])
def test_small_palettes_are_globally_optimal(depth: int) -> None:
    """Brute-force check that no candidate subset has a larger minimum distance."""
    cands = pal.candidate_colours()
    best = max(pal.min_pairwise_distance(cands[list(s)]) for s in itertools.combinations(range(len(cands)), depth))
    assert pal.min_pairwise_distance(pal.palette(depth)) == pytest.approx(best)


def test_distance_shrinks_with_depth() -> None:
    dists = [pal.min_pairwise_distance(pal.palette(d)) for d in (2, 4, 8, 16)]
    assert dists == sorted(dists, reverse=True)


def test_unknown_depth_rejected() -> None:
    with pytest.raises(ValueError):
        pal.palette(3)
