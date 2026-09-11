"""Hard requirement 1: fiducials are constant across all configurations."""

from __future__ import annotations

import hashlib
import itertools

import cv2
import numpy as np
import pytest

from prism_share.codec import layout
from prism_share.codec.encoder import encode
from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    FIDUCIAL_BITS,
    FIDUCIAL_DICTIONARY,
    FIDUCIAL_IDS,
    FIDUCIAL_PATTERNS,
    KEEPOUT_PX,
    SWEEP_CELL_PX,
    CodecParams,
)

FRAME_PX = CodecParams().frame_px
ALL_CONFIGS = [
    CodecParams(colour_depth=d, cell_px=c) for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX)
]
# Pinned hash of the static region (border band + keep-out squares with markers)
# at frame_px = 1024. If this changes, every capture made so far is invalidated.
STATIC_CANVAS_SHA256 = "100c49b588fb7c9c58deabf051f649fbc923a2598978ef5e3ad4065d8b6b9a9b"


def test_static_region_is_pinned() -> None:
    assert hashlib.sha256(layout.base_canvas(FRAME_PX).tobytes()).hexdigest() == STATIC_CANVAS_SHA256


VARIANTS = [
    *ALL_CONFIGS,
    CodecParams(seed=99),
    CodecParams(ecc_data=200, ecc_total=255),
    CodecParams(cell_gap_px=0, cell_px=4),
    CodecParams(cell_gap_px=2, colour_depth=16),
]


@pytest.mark.parametrize("params", VARIANTS, ids=lambda p: p.label)
def test_static_region_identical_across_all_configurations(params: CodecParams) -> None:
    mask = layout.static_mask(FRAME_PX)
    frame = encode(b"payload that differs per config " + params.label.encode(), params, n_frames=1)[0]
    assert np.array_equal(frame[mask], layout.base_canvas(FRAME_PX)[mask])


def test_static_region_is_pure_black_and_white() -> None:
    static = layout.base_canvas(FRAME_PX)[layout.static_mask(FRAME_PX)]
    assert set(np.unique(static)) == {0, 255}
    assert (static == static[:, :1]).all(), "static region must be achromatic (R == G == B)"


def test_fiducial_patterns_match_opencv_dictionary() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, FIDUCIAL_DICTIONARY))
    assert dictionary.markerSize == FIDUCIAL_BITS
    for marker_id, pattern in zip(FIDUCIAL_IDS, FIDUCIAL_PATTERNS, strict=True):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, FIDUCIAL_BITS + 2, borderBits=1)
        bits = img[1:-1, 1:-1] > 127
        assert ["".join("1" if b else "0" for b in row) for row in bits] == list(pattern)


@pytest.mark.parametrize("params", [ALL_CONFIGS[0], ALL_CONFIGS[-1]], ids=lambda p: p.label)
def test_opencv_finds_all_fiducials_where_layout_says(params: CodecParams) -> None:
    frame = encode(b"x", params, n_frames=1)[0]
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, FIDUCIAL_DICTIONARY)))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    assert ids is not None and sorted(ids.ravel().tolist()) == sorted(FIDUCIAL_IDS)
    expected = layout.fiducial_corners(FRAME_PX)
    for marker_corners, marker_id in zip(corners, ids.ravel(), strict=True):
        slot = FIDUCIAL_IDS.index(int(marker_id))
        # OpenCV reports corners on pixel centres of the boundary, within ~1 px of the ideal edges.
        assert np.abs(marker_corners[0] - expected[slot]).max() <= 1.0


@pytest.mark.parametrize("params", ALL_CONFIGS, ids=lambda p: p.label)
def test_cells_are_inside_data_region_and_disjoint(params: CodecParams) -> None:
    rows, cols = layout.cell_pixel_index(params)
    occupancy = np.zeros((FRAME_PX, FRAME_PX), dtype=np.int64)
    np.add.at(occupancy, (rows.ravel(), cols.ravel()), 1)
    assert occupancy.max() == 1, "cells overlap"
    assert not (occupancy.astype(bool) & layout.static_mask(FRAME_PX)).any(), "cell in static region"

    # Every cell keeps cell_gap_px of background from the static region.
    g = params.cell_gap_px
    grown = np.zeros_like(occupancy, dtype=bool)
    lay = layout.grid_layout(params)
    for x, y in zip(lay.cell_x, lay.cell_y, strict=True):
        grown[y - g : y + params.cell_px + g, x - g : x + params.cell_px + g] = True
    assert not (grown & layout.static_mask(FRAME_PX)).any()


def test_cell_order_is_row_major() -> None:
    lay = layout.grid_layout(CodecParams())
    keys = list(zip(lay.cell_y.tolist(), lay.cell_x.tolist(), strict=True))
    assert keys == sorted(keys)


def test_reference_masks_sample_the_right_levels() -> None:
    canvas = layout.base_canvas(FRAME_PX)
    assert (canvas[layout.white_reference_mask(FRAME_PX)] == 255).all()
    assert (canvas[layout.black_reference_mask(FRAME_PX)] == 0).all()
    assert layout.white_reference_mask(FRAME_PX).sum() > 1000
    assert layout.black_reference_mask(FRAME_PX).sum() > 1000
    assert not (layout.white_reference_mask(FRAME_PX) & ~layout.static_mask(FRAME_PX)).any()
    assert not (layout.black_reference_mask(FRAME_PX) & ~layout.static_mask(FRAME_PX)).any()


def test_keepout_squares_are_in_the_corners() -> None:
    far = FRAME_PX - KEEPOUT_PX
    assert layout.keepout_origins(FRAME_PX) == ((0, 0), (far, 0), (far, far), (0, far))
