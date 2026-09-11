"""Colour palettes for colour depths 1, 2, 4, 8 and 16.

Palettes are ink colours drawn on the black data-region background.

* Depth 1 (monochrome) is ``MONO_INK_RGB`` (white).
* Depth D >= 2 is chosen by *exhaustive* search over all D-subsets of the
  candidate set: RGB triples with every channel in ``PALETTE_LEVELS`` and at
  least one channel at full scale (255). Full scale in some channel means every
  palette colour has the same HSV value, so shape contrast against the black
  background is identical for every colour (the decoder reads shape from the
  per-pixel channel maximum).
* Objective, lexicographic and exact (integer arithmetic):
    1. maximise the minimum pairwise squared Euclidean distance in 8-bit code
       values,
    2. then maximise the minimum Rec. 709 luma (better sensor SNR),
    3. then minimise the number of pairs at the minimum distance,
    4. then maximise the sum of pairwise squared distances,
    5. then the lexicographically first subset in candidate order.

Each depth is optimised independently (palettes are not nested): each colour
condition gets the best palette this rule can find, which is the fair test of
a hypothesis that predicts colour will *not* pay for itself.

Palette order fixes the colour index -> RGB mapping; within a palette colours
are sorted by candidate order (R-major, ascending code values).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import numpy.typing as npt

from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    LUMA_WEIGHTS_709,
    MONO_INK_RGB,
    PALETTE_LEVELS,
    CodecParams,
)

UInt8Array = npt.NDArray[np.uint8]


def palette_for(params: CodecParams) -> UInt8Array:
    """Ink colours for ``params``: (colour_depth, 3) uint8 RGB."""
    return palette(params.colour_depth)


@functools.lru_cache(maxsize=None)
def palette(colour_depth: int) -> UInt8Array:
    """Ink colours for ``colour_depth``: (colour_depth, 3) uint8 RGB, read-only."""
    if colour_depth not in ALLOWED_COLOUR_DEPTHS:
        raise ValueError(f"no palette for colour_depth={colour_depth}")
    if colour_depth == 1:
        out = np.array([MONO_INK_RGB], dtype=np.uint8)
    else:
        cands = candidate_colours()
        out = cands[list(_best_subset(cands, colour_depth))]
    out.setflags(write=False)
    return out


def candidate_colours() -> UInt8Array:
    """All RGB triples over PALETTE_LEVELS with at least one full-scale channel."""
    top = max(PALETTE_LEVELS)
    triples = [c for c in itertools.product(PALETTE_LEVELS, repeat=3) if max(c) == top]
    return np.array(triples, dtype=np.uint8)


def min_pairwise_distance(colours: UInt8Array) -> float:
    """Minimum pairwise Euclidean distance in 8-bit code values (inf for one colour)."""
    if len(colours) < 2:
        return float("inf")
    sq = _squared_distances(colours.astype(np.int64))
    return float(np.sqrt(sq[np.triu_indices(len(colours), k=1)].min()))


def luma_709(colours: UInt8Array) -> npt.NDArray[np.float64]:
    """Rec. 709 luma of each colour on the 0-255 code-value scale."""
    weights = np.array(LUMA_WEIGHTS_709, dtype=np.float64)
    return colours.astype(np.float64) @ weights / weights.sum()


def _squared_distances(colours: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    diff = colours[:, None, :] - colours[None, :, :]
    return (diff * diff).sum(axis=-1)


def _best_subset(cands: UInt8Array, depth: int) -> tuple[int, ...]:
    ints = cands.astype(np.int64)
    sq = _squared_distances(ints)
    luma = ints @ np.array(LUMA_WEIGHTS_709, dtype=np.int64)
    subsets = np.array(list(itertools.combinations(range(len(cands)), depth)), dtype=np.int64)
    ii, jj = np.triu_indices(depth, k=1)
    pair_sq = sq[subsets[:, ii], subsets[:, jj]]  # (n_subsets, n_pairs)
    min_sq = pair_sq.min(axis=1)
    min_luma = luma[subsets].min(axis=1)
    at_min = (pair_sq == min_sq[:, None]).sum(axis=1)
    total = pair_sq.sum(axis=1)
    # np.lexsort sorts ascending with the LAST key primary; the subset index
    # (combinations are generated in lexicographic order) is the final tie-break.
    order = np.lexsort((np.arange(len(subsets)), -total, at_min, -min_luma, -min_sq))
    return tuple(int(i) for i in subsets[order[0]])
