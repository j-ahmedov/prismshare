"""The reference frame: the experimental control the capture app locks onto.

The capture app shows this frame first and locks auto-exposure, auto-focus and
auto-white-balance on it before the code frames play. It therefore has to:

* be **geometrically identical** to the codes: same ``frame_px``, same static
  region (white border, keep-out squares, ArUco fiducials), taken verbatim
  from ``layout.base_canvas``, and the same index band, carrying the reserved
  index 0 so a capture of it is identified structurally;
* present the camera with the **same linear-light average** as the codes.
  Auto-exposure meters linear light, not code values: a flat sRGB 128 grey is
  0.216 linear, while a 50/50 black-and-white pattern is 0.5. Matching the
  wrong quantity would leave every code frame about a stop off;
* contain only pure black and pure white (no colour for AWB to be pulled by,
  no mid-grey whose emitted light depends on the panel's transfer curve), with
  fine high-contrast texture for contrast-detect autofocus;
* be **deterministic and identical for every run**.

Construction: the data region is an ordered-dither (Bayer) pattern of
``REFERENCE_DITHER_BLOCK_PX`` square blocks. Blocks are ranked by their dither
threshold (ties by block index), and exactly the number of white blocks that
brings the *whole-frame* mean linear luminance closest to the target is
turned white.

One reference cannot match every configuration: the codes span about one stop
(monochrome, with white ink, is brightest). The default target
``REFERENCE_TARGET_LINEAR_MEAN`` is pinned at the geometric midpoint of the
sweep, so the worst mismatch is about +/-0.51 stops, and every configuration
is captured with *identical* camera settings - the control. The mismatch for
every configuration is computed by ``configuration_luminance`` and recorded by
the CLI::

    python -m prism_share.transmit.reference --out data/reference.png --report docs/reference_frame.md

A per-configuration reference (``--match colour_depth,cell_px``) is available
but is not the default: it would give each configuration different camera
exposure, confounding exposure with the parameter under test.
"""

from __future__ import annotations

import argparse
import functools
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from prism_share.codec.encoder import encode_frames, write_png
from prism_share.codec.framing import frame_capacity
from prism_share.codec.layout import base_canvas, draw_index_band, index_band_mask, static_mask
from prism_share.codec.params import (
    ALLOWED_COLOUR_DEPTHS,
    INDEX_BAND_REFERENCE,
    REFERENCE_DITHER_BLOCK_PX,
    REFERENCE_DITHER_SIZE,
    REFERENCE_TARGET_LINEAR_MEAN,
    SWEEP_CELL_PX,
    WHITE_RGB,
    CodecParams,
)
from prism_share.codec.prng import keystream
from prism_share.colourspace import linear_luminance, mean_linear_luminance

UInt8Array = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int64]

#: Frames averaged per configuration when measuring code luminance. Whitening
#: makes frames statistically identical; the spread between frames is reported.
LUMINANCE_FRAMES = 3


def bayer_matrix(size: int) -> IntArray:
    """Ordered-dither threshold matrix with values 0 .. size**2 - 1 (size a power of two)."""
    if size < 1 or size & (size - 1):
        raise ValueError("size must be a power of two")
    base = np.array([[0, 2], [3, 1]], dtype=np.int64)
    m = np.zeros((1, 1), dtype=np.int64)
    while m.shape[0] < size:
        m = base.size * np.tile(m, base.shape) + np.kron(base, np.ones_like(m))
    return m


@functools.lru_cache(maxsize=None)
def reference_frame(frame_px: int = CodecParams().frame_px, target: float = REFERENCE_TARGET_LINEAR_MEAN) -> UInt8Array:
    """The reference frame for ``frame_px``: (frame_px, frame_px, 3) uint8 RGB, read-only."""
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    block = REFERENCE_DITHER_BLOCK_PX
    if frame_px % block:
        raise ValueError(f"frame_px must be a multiple of {block}")
    canvas = np.array(base_canvas(frame_px))
    draw_index_band(canvas, INDEX_BAND_REFERENCE, frame_px)
    # The band is reserved: the dither may not use it, and it counts as static
    # luminance (index 0 is all-black) when solving for the white fraction.
    static = static_mask(frame_px) | index_band_mask(frame_px)

    # Blocks of the data region (the static region is block-aligned: all its edges are even).
    blocks_per_side = frame_px // block
    block_static = static.reshape(blocks_per_side, block, blocks_per_side, block).any(axis=(1, 3))
    by, bx = np.nonzero(~block_static)
    dither = bayer_matrix(REFERENCE_DITHER_SIZE)
    threshold = dither[by % REFERENCE_DITHER_SIZE, bx % REFERENCE_DITHER_SIZE]
    order = np.lexsort((np.arange(len(by)), threshold))  # rank by threshold, ties by block index

    # Whole-frame mean = (static luminance + white data pixels) / all pixels; pick the count closest to target.
    static_sum = float(linear_luminance(canvas)[static].sum())
    pixels_per_block = block * block
    n_white = round((target * frame_px * frame_px - static_sum) / pixels_per_block)
    n_white = int(np.clip(n_white, 0, len(order)))

    white_blocks = np.zeros((blocks_per_side, blocks_per_side), dtype=bool)
    white_blocks[by[order[:n_white]], bx[order[:n_white]]] = True
    white_pixels = np.kron(white_blocks, np.ones((block, block), dtype=bool)) & ~static
    canvas[white_pixels] = WHITE_RGB
    canvas.setflags(write=False)
    return canvas


@dataclass(frozen=True)
class LuminanceRow:
    colour_depth: int
    cell_px: int
    frame_mean: float
    """Whole-frame mean linear luminance, averaged over LUMINANCE_FRAMES frames."""
    data_mean: float
    """Same, data region only."""
    frame_spread: float
    """Max - min of frame_mean across the frames (content dependence)."""
    stops_vs_reference: float
    """log2(frame_mean / reference mean): + = code brighter than the reference."""


def configuration_luminance(params: CodecParams, reference_mean: float) -> LuminanceRow:
    """Linear-light statistics of real encoded frames of ``params``."""
    payload = keystream(params.seed, "luminance-payload", frame_capacity(params).block_bytes * LUMINANCE_FRAMES)
    frames = encode_frames(payload, params, n_frames=LUMINANCE_FRAMES)
    data = ~static_mask(params.frame_px)
    means = [mean_linear_luminance(f.image) for f in frames]
    data_means = [float(linear_luminance(f.image)[data].mean()) for f in frames]
    frame_mean = float(np.mean(means))
    return LuminanceRow(
        colour_depth=params.colour_depth,
        cell_px=params.cell_px,
        frame_mean=frame_mean,
        data_mean=float(np.mean(data_means)),
        frame_spread=float(np.ptp(means)),
        stops_vs_reference=float(np.log2(frame_mean / reference_mean)),
    )


def luminance_table(reference: UInt8Array, configs: list[CodecParams]) -> list[LuminanceRow]:
    ref_mean = mean_linear_luminance(reference)
    return [configuration_luminance(p, ref_mean) for p in configs]


def sweep_configurations(frame_px: int = CodecParams().frame_px) -> list[CodecParams]:
    """Every configuration of the sweep (colour depth x cell size)."""
    return [CodecParams(colour_depth=d, cell_px=c, frame_px=frame_px)
            for d, c in itertools.product(ALLOWED_COLOUR_DEPTHS, SWEEP_CELL_PX)]


def extreme_configurations(configs: list[CodecParams] | None = None) -> tuple[LuminanceRow, LuminanceRow]:
    """(brightest, darkest) configuration by whole-frame mean linear luminance.

    With exposure locked on the reference, these two are the ones that clip
    first (brightest, at the highlights) and sink furthest into the noise floor
    (darkest). The reference passing a clip check proves neither.
    """
    rows = luminance_table(reference_frame(), configs or sweep_configurations())
    return max(rows, key=lambda r: r.frame_mean), min(rows, key=lambda r: r.frame_mean)


def _format(reference: UInt8Array, rows: list[LuminanceRow], target: float) -> str:
    ref_mean = mean_linear_luminance(reference)
    ref_data = float(linear_luminance(reference)[~static_mask(reference.shape[0])].mean())
    white_share = float((reference[~static_mask(reference.shape[0])] == max(WHITE_RGB)).all(axis=1).mean())
    worst = max(rows, key=lambda r: abs(r.stops_vs_reference))
    lines = [
        f"Reference frame: target {target:.4f}, achieved whole-frame mean linear luminance {ref_mean:.4f} "
        f"(data region {ref_data:.4f}, {100 * white_share:.2f} % white).",
        f"Worst mismatch: {worst.stops_vs_reference:+.3f} stops (depth {worst.colour_depth}, {worst.cell_px} px).",
        "",
        "| colour depth | cell px | frame mean (linear) | data-region mean | frame-to-frame spread | stops vs reference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.colour_depth} | {r.cell_px} | {r.frame_mean:.4f} | {r.data_mean:.4f} | {r.frame_spread:.5f} | {r.stops_vs_reference:+.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the reference frame and report linear-light mismatches.")
    parser.add_argument("--out", default="data/reference.png", help="PNG to write")
    parser.add_argument("--report", default=None, help="optional Markdown file recording the table")
    parser.add_argument("--frame-px", type=int, default=CodecParams().frame_px)
    parser.add_argument("--target", type=float, default=REFERENCE_TARGET_LINEAR_MEAN)
    parser.add_argument(
        "--match", default=None, metavar="DEPTH,CELL_PX",
        help="build a per-configuration reference matched to this configuration instead of the pinned target",
    )
    args = parser.parse_args(argv)

    configs = sweep_configurations(args.frame_px)
    target = args.target
    if args.match:
        depth, cell = (int(v) for v in args.match.split(","))
        target = configuration_luminance(CodecParams(colour_depth=depth, cell_px=cell, frame_px=args.frame_px), 1.0).frame_mean
    reference = reference_frame(args.frame_px, target)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_png(args.out, reference)
    text = _format(reference, luminance_table(reference, configs), target)
    print(text)
    print(f"\nwrote {args.out}")
    if args.report:
        Path(args.report).write_text(
            "# Reference frame luminance\n\n"
            "Generated by `python -m prism_share.transmit.reference"
            + (f" --report {args.report}" if args.report else "")
            + "`. Linear luminance assumes an sRGB panel (IEC 61966-2-1 transfer, Rec. 709 primaries); "
            "it predicts, it does not measure, the light the panel emits.\n\n" + text + "\n"
        )
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
