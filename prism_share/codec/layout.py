"""Frame geometry: fiducials, static region and the data-cell grid.

Frame anatomy (square, ``frame_px`` side, RGB, origin top-left, x right, y down):

* A white border ``BORDER_PX`` wide around the whole frame.
* At each corner a ``KEEPOUT_PX`` square, white, holding an ArUco marker
  (black on white, ``FIDUCIAL_PX`` side) at offset ``BORDER_PX`` from both
  frame edges. Marker ids ``FIDUCIAL_IDS`` run clockwise from top-left.
* A reserved **index band** of large black/white blocks between the two top
  keep-out squares, carrying the frame's own index (see ``index_band``).
* Everything else is the data region: black background carrying the cells.

The border band plus the four keep-out squares form the **static region**. Its
pixels depend on ``frame_px`` alone - never on colour depth, cell size, ECC or
seed - and contain only pure black and pure white. This is the constraint that
keeps detection difficulty out of the experiment.

The index band is *not* part of the static region: its geometry is constant
for every configuration, but its blocks carry the frame index, so two frames
of the same configuration differ there. It never uses colour.

The cell grid: pitch = cell_px + cell_gap_px, as many cells per side as fit in
the data region with at least ``cell_gap_px`` of background on every side of
every cell; the grid is centred. Cells whose gap-padded box would touch a
keep-out square or the index band are dropped. Remaining cells are numbered row-major (top to
bottom, left to right); that number is the cell index everywhere else.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prism_share.codec.params import (
    BACKGROUND_RGB,
    BLACK_REF_INSET_PX,
    BLACK_RGB,
    BORDER_PX,
    INDEX_BAND_BITS,
    INDEX_BAND_BLOCK_PX,
    INDEX_BAND_MARGIN_PX,
    INDEX_BAND_REPEATS,
    FIDUCIAL_BORDER_BITS,
    FIDUCIAL_IDS,
    FIDUCIAL_MODULE_PX,
    FIDUCIAL_PATTERNS,
    FIDUCIAL_PX,
    KEEPOUT_PX,
    WHITE_REF_INSET_PX,
    WHITE_RGB,
    CodecParams,
)

UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class GridLayout:
    """Positions of the data cells for one CodecParams."""

    cells_per_side: int
    """Grid columns (= rows) before dropping keep-out cells."""
    origin_px: int
    """x (= y) of the top-left pixel of grid cell (0, 0)."""
    cell_x: IntArray
    """(n_cells,) x of each used cell's top-left pixel, in cell-index order."""
    cell_y: IntArray
    """(n_cells,) y of each used cell's top-left pixel, in cell-index order."""

    @property
    def n_cells(self) -> int:
        return len(self.cell_x)


# --------------------------------------------------------------------------- #
# Fiducials and static region
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def fiducial_marker(slot: int) -> UInt8Array:
    """Marker image for corner ``slot`` (0=TL, 1=TR, 2=BR, 3=BL): (FIDUCIAL_PX,)*2, 0/255."""
    interior = np.array([[ch == "1" for ch in row] for row in FIDUCIAL_PATTERNS[slot]], dtype=bool)
    modules = np.pad(interior, FIDUCIAL_BORDER_BITS, constant_values=False)
    block = np.ones((FIDUCIAL_MODULE_PX, FIDUCIAL_MODULE_PX), dtype=np.uint8)
    marker = np.kron(modules.astype(np.uint8), block) * np.uint8(max(WHITE_RGB))
    marker.setflags(write=False)
    return marker


def fiducial_origins(frame_px: int) -> tuple[tuple[int, int], ...]:
    """(x, y) of each marker's top-left pixel, in FIDUCIAL_IDS slot order."""
    far = frame_px - BORDER_PX - FIDUCIAL_PX
    return ((BORDER_PX, BORDER_PX), (far, BORDER_PX), (far, far), (BORDER_PX, far))


def fiducial_corners(frame_px: int) -> npt.NDArray[np.float64]:
    """Ideal marker corners in the rectified frame, (n_markers, 4, 2) float (x, y).

    Corner order follows OpenCV ArUco: TL, TR, BR, BL of each marker, as
    continuous pixel-edge coordinates (a marker at x=16 spanning 72 px has
    corners at x=16.0 and x=88.0).
    """
    out = []
    for x, y in fiducial_origins(frame_px):
        out.append([[x, y], [x + FIDUCIAL_PX, y], [x + FIDUCIAL_PX, y + FIDUCIAL_PX], [x, y + FIDUCIAL_PX]])
    return np.array(out, dtype=np.float64)


def keepout_origins(frame_px: int) -> tuple[tuple[int, int], ...]:
    """(x, y) of each keep-out square's top-left pixel, in slot order."""
    far = frame_px - KEEPOUT_PX
    return ((0, 0), (far, 0), (far, far), (0, far))


@functools.lru_cache(maxsize=None)
def static_mask(frame_px: int) -> BoolArray:
    """True on the static region (border band + keep-out squares)."""
    mask = np.zeros((frame_px, frame_px), dtype=bool)
    mask[:BORDER_PX, :] = True
    mask[-BORDER_PX:, :] = True
    mask[:, :BORDER_PX] = True
    mask[:, -BORDER_PX:] = True
    for x, y in keepout_origins(frame_px):
        mask[y : y + KEEPOUT_PX, x : x + KEEPOUT_PX] = True
    mask.setflags(write=False)
    return mask


@functools.lru_cache(maxsize=None)
def white_reference_mask(frame_px: int) -> BoolArray:
    """Pixels that are always white and far from any edge: the middle half of the border band.

    Used by the decoder as its white level. Excludes the keep-out squares so a
    marker's black modules can never bleed in.
    """
    mask = np.zeros((frame_px, frame_px), dtype=bool)
    lo, hi = WHITE_REF_INSET_PX, BORDER_PX - WHITE_REF_INSET_PX
    span = slice(KEEPOUT_PX, frame_px - KEEPOUT_PX)
    mask[lo:hi, span] = True
    mask[frame_px - hi : frame_px - lo, span] = True
    mask[span, lo:hi] = True
    mask[span, frame_px - hi : frame_px - lo] = True
    mask.setflags(write=False)
    return mask


@functools.lru_cache(maxsize=None)
def black_reference_mask(frame_px: int) -> BoolArray:
    """Pixels that are always black: the central half of each marker's border modules.

    Used by the decoder as its black level. Taken from the fiducials (not the
    data region) so it is identical for every configuration.
    """
    mask = np.zeros((frame_px, frame_px), dtype=bool)
    ring = np.zeros((FIDUCIAL_PX, FIDUCIAL_PX), dtype=bool)
    inset = BLACK_REF_INSET_PX
    border = FIDUCIAL_BORDER_BITS * FIDUCIAL_MODULE_PX
    ring[inset : FIDUCIAL_PX - inset, inset : FIDUCIAL_PX - inset] = True
    ring[border - inset : FIDUCIAL_PX - border + inset, border - inset : FIDUCIAL_PX - border + inset] = False
    for x, y in fiducial_origins(frame_px):
        mask[y : y + FIDUCIAL_PX, x : x + FIDUCIAL_PX] = ring
    mask.setflags(write=False)
    return mask


@functools.lru_cache(maxsize=None)
def base_canvas(frame_px: int) -> UInt8Array:
    """Frame with the static region drawn and an empty (background) data region."""
    canvas = np.empty((frame_px, frame_px, len(WHITE_RGB)), dtype=np.uint8)
    canvas[...] = BACKGROUND_RGB
    canvas[static_mask(frame_px)] = WHITE_RGB
    for slot, (x, y) in enumerate(fiducial_origins(frame_px)):
        marker = fiducial_marker(slot)
        region = canvas[y : y + FIDUCIAL_PX, x : x + FIDUCIAL_PX]
        region[marker == 0] = BLACK_RGB
        region[marker != 0] = WHITE_RGB
    canvas.setflags(write=False)
    return canvas


# --------------------------------------------------------------------------- #
# Index band
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexBand:
    """Where the frame-index band sits and how big its blocks are. Identical for every CodecParams."""

    x: int
    y: int
    block_px: int
    blocks: int
    """INDEX_BAND_BITS * INDEX_BAND_REPEATS."""

    @property
    def width(self) -> int:
        return self.blocks * self.block_px

    @property
    def height(self) -> int:
        return self.block_px

    def block_origin(self, position: int) -> tuple[int, int]:
        return self.x + position * self.block_px, self.y


@functools.lru_cache(maxsize=None)
def index_band(frame_px: int) -> IndexBand:
    """The band: one row of blocks, centred between the two top keep-out squares."""
    blocks = INDEX_BAND_BITS * INDEX_BAND_REPEATS
    width = blocks * INDEX_BAND_BLOCK_PX
    available = frame_px - 2 * KEEPOUT_PX
    if width > available:
        raise ValueError(f"index band ({width} px) does not fit between the keep-out squares ({available} px)")
    return IndexBand(x=(frame_px - width) // 2, y=BORDER_PX + INDEX_BAND_MARGIN_PX,
                     block_px=INDEX_BAND_BLOCK_PX, blocks=blocks)


def index_band_bits(index: int) -> list[int]:
    """The blocks of ``index``: its bits MSB-first, repeated INDEX_BAND_REPEATS times."""
    if not 0 <= index < 2**INDEX_BAND_BITS:
        raise ValueError(f"frame index must fit in {INDEX_BAND_BITS} bits, got {index}")
    bits = [(index >> shift) & 1 for shift in range(INDEX_BAND_BITS - 1, -1, -1)]
    return bits * INDEX_BAND_REPEATS


@functools.lru_cache(maxsize=None)
def index_band_mask(frame_px: int) -> BoolArray:
    """True on the band's blocks (not its background margin)."""
    band = index_band(frame_px)
    mask = np.zeros((frame_px, frame_px), dtype=bool)
    mask[band.y : band.y + band.height, band.x : band.x + band.width] = True
    mask.setflags(write=False)
    return mask


def draw_index_band(canvas: UInt8Array, index: int, frame_px: int) -> None:
    """Draw ``index`` into ``canvas`` in place: pure black and white blocks, never colour."""
    band = index_band(frame_px)
    for position, bit in enumerate(index_band_bits(index)):
        x, y = band.block_origin(position)
        canvas[y : y + band.height, x : x + band.block_px] = WHITE_RGB if bit else BLACK_RGB


# --------------------------------------------------------------------------- #
# Data grid
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def grid_layout(params: CodecParams) -> GridLayout:
    """Cell positions for ``params`` (cached; arrays are read-only)."""
    return _grid(params, reserve_band=True)


@functools.lru_cache(maxsize=None)
def band_cell_cost(params: CodecParams) -> int:
    """Data cells the index band displaces for ``params``.

    The band is measurement apparatus, not codec: a deployed system would carry
    its frame index in the fountain header. Its cost depends on cell_px, so it
    is reported per configuration and credited back in a second goodput column.
    """
    return _grid(params, reserve_band=False).n_cells - grid_layout(params).n_cells


def _grid(params: CodecParams, *, reserve_band: bool) -> GridLayout:
    inner = params.frame_px - 2 * BORDER_PX
    gap, pitch = params.cell_gap_px, params.pitch_px
    per_side = (inner - gap) // pitch
    extent = per_side * pitch + gap  # grid including the outer gaps
    origin = BORDER_PX + (inner - extent) // 2 + gap

    idx = np.arange(per_side, dtype=np.int64)
    ys, xs = np.meshgrid(origin + idx * pitch, origin + idx * pitch, indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()

    # A cell is kept if its box grown by `gap` on every side misses every reserved
    # rectangle: the four keep-out squares and the index band.
    reserved = [(kx, ky, KEEPOUT_PX, KEEPOUT_PX) for kx, ky in keepout_origins(params.frame_px)]
    if reserve_band:
        band = index_band(params.frame_px)
        reserved.append((band.x, band.y, band.width, band.height))
    keep = np.ones(len(xs), dtype=bool)
    for rx, ry, rw, rh in reserved:
        overlap_x = (xs - gap < rx + rw) & (xs + params.cell_px + gap > rx)
        overlap_y = (ys - gap < ry + rh) & (ys + params.cell_px + gap > ry)
        keep &= ~(overlap_x & overlap_y)

    cell_x, cell_y = xs[keep], ys[keep]
    cell_x.setflags(write=False)
    cell_y.setflags(write=False)
    return GridLayout(cells_per_side=per_side, origin_px=origin, cell_x=cell_x, cell_y=cell_y)


def cell_pixel_index(params: CodecParams) -> tuple[IntArray, IntArray]:
    """Row and column index arrays, each (n_cells, cell_px, cell_px), for fancy indexing.

    ``frame[rows, cols]`` gathers every cell's pixels as (n_cells, cell_px, cell_px[, 3]).
    """
    layout = grid_layout(params)
    offs = np.arange(params.cell_px, dtype=np.int64)
    rows = layout.cell_y[:, None, None] + offs[None, :, None]
    cols = layout.cell_x[:, None, None] + offs[None, None, :]
    rows, cols = np.broadcast_arrays(rows, cols)
    return rows, cols
