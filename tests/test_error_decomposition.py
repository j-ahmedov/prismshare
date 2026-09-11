"""Hard requirement 6: shape errors and colour errors are measured independently.

Injects synthetic faults of one kind only and checks that they show up in that
kind's symbol error rate only, exactly N / n_cells. If the decoder could not
separate a pure colour fault from a pure shape fault, the shape/colour SER
decomposition could not test the hypothesis.

This validates the decoder's decisions; when metrics.py exists (step 6) the
same injected frames must give the same numbers through its SER functions.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from prism_share.codec.decoder import read_symbols
from prism_share.codec.encoder import encode_frames, render_frame
from prism_share.codec.params import ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX, CodecParams

N_FAULTS = 500
GRID = [CodecParams(colour_depth=d, cell_px=c) for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX)]


def _ser(params: CodecParams, fault: str) -> tuple[float, float, float, float]:
    """(shape SER, colour SER, expected shape SER, expected colour SER) after injection."""
    truth = encode_frames(b"decomposition", params, n_frames=1)[0]
    rng = np.random.default_rng(params.cell_px * 100 + params.colour_depth)
    n_cells = len(truth.glyphs)
    cells = rng.choice(n_cells, size=N_FAULTS, replace=False)
    glyphs, colours = truth.glyphs.copy(), truth.colours.copy()
    n_shape = n_colour = 0
    if fault in ("colour", "both"):
        sel = cells if fault == "colour" else cells[: N_FAULTS // 2]
        # A non-zero offset mod depth always lands on a *different* palette entry.
        colours[sel] = (colours[sel] + rng.integers(1, params.colour_depth, size=len(sel))) % params.colour_depth
        n_colour = len(sel)
    if fault in ("shape", "both"):
        sel = cells if fault == "shape" else cells[N_FAULTS // 2 :]
        glyphs[sel] = (glyphs[sel] + rng.integers(1, params.glyph_count, size=len(sel))) % params.glyph_count
        n_shape = len(sel)
    readout = read_symbols(render_frame(glyphs, colours, params), params)
    return (
        float(np.mean(readout.glyphs != truth.glyphs)),
        float(np.mean(readout.colours != truth.colours)),
        n_shape / n_cells,
        n_colour / n_cells,
    )


@pytest.mark.parametrize("params", [p for p in GRID if p.colour_depth > 1], ids=lambda p: p.label)
def test_pure_colour_fault_is_pure_colour_error(params: CodecParams) -> None:
    shape, colour, exp_shape, exp_colour = _ser(params, "colour")
    assert shape == 0.0
    assert colour == exp_colour


@pytest.mark.parametrize("params", GRID, ids=lambda p: p.label)
def test_pure_shape_fault_is_pure_shape_error(params: CodecParams) -> None:
    shape, colour, exp_shape, exp_colour = _ser(params, "shape")
    assert shape == exp_shape
    assert colour == 0.0


@pytest.mark.parametrize("params", [CodecParams(colour_depth=16, cell_px=4), CodecParams(colour_depth=2, cell_px=10)], ids=lambda p: p.label)
def test_mixed_faults_split_exactly(params: CodecParams) -> None:
    shape, colour, exp_shape, exp_colour = _ser(params, "both")
    assert (shape, colour) == (exp_shape, exp_colour)
