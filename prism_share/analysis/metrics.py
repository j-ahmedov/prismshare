"""Error metrics against ground truth, and derived goodput.

Shape errors and colour errors are reported **separately** (hard requirement
6): a cell has a shape error when its decoded glyph differs from the truth and
a colour error when its decoded colour differs. A cell can have both. The
decoder makes the two decisions independently, so the two rates measure
different physical effects. ``symbol_ser`` (either wrong) is provided for
completeness but is never a substitute for the pair.

Goodput is derived, never timed (hard requirement 7)::

    goodput = payload_bytes_per_frame * frame_yield * ASSUMED_FPS
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from prism_share.codec.framing import symbols_to_stream
from prism_share.codec.params import ASSUMED_FPS, BITS_PER_BYTE, CodecParams

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SymbolErrors:
    """Per-cell error flags of one frame, in cell-index order."""

    shape: BoolArray
    colour: BoolArray

    @property
    def n_cells(self) -> int:
        return len(self.shape)

    @property
    def shape_ser(self) -> float:
        """Fraction of cells whose glyph is wrong."""
        return float(self.shape.mean())

    @property
    def colour_ser(self) -> float:
        """Fraction of cells whose colour is wrong (0 for monochrome by construction)."""
        return float(self.colour.mean())

    @property
    def symbol_ser(self) -> float:
        """Fraction of cells with a shape error, a colour error, or both."""
        return float((self.shape | self.colour).mean())


def symbol_errors(
    truth_glyphs: npt.ArrayLike,
    truth_colours: npt.ArrayLike,
    decoded_glyphs: npt.ArrayLike,
    decoded_colours: npt.ArrayLike,
) -> SymbolErrors:
    tg, tc = np.asarray(truth_glyphs), np.asarray(truth_colours)
    dg, dc = np.asarray(decoded_glyphs), np.asarray(decoded_colours)
    if not tg.shape == tc.shape == dg.shape == dc.shape:
        raise ValueError("truth and decoded symbol arrays must have equal shapes")
    return SymbolErrors(shape=tg != dg, colour=tc != dc)


def byte_errors(
    truth_glyphs: npt.ArrayLike,
    truth_colours: npt.ArrayLike,
    decoded_glyphs: npt.ArrayLike,
    decoded_colours: npt.ArrayLike,
    params: CodecParams,
) -> BoolArray:
    """(capacity_bytes,) True where the decoded on-screen stream byte differs from the truth.

    Computed by packing both symbol sets through the framing layer and
    comparing bytes, so it is exactly the byte error pattern an RS decoder
    would see (whitening cancels in the comparison).
    """
    truth = np.frombuffer(symbols_to_stream(np.asarray(truth_glyphs), np.asarray(truth_colours), params), np.uint8)
    got = np.frombuffer(symbols_to_stream(np.asarray(decoded_glyphs), np.asarray(decoded_colours), params), np.uint8)
    return truth != got


class FrameOutcome(str, Enum):
    """What happened to one captured code frame. Always three outcomes, never two:
    they have different causes (optics/geometry vs channel errors) and different fixes."""

    DETECTION_FAILED = "detection_failed"
    NOT_RECOVERABLE = "not_recoverable"
    RECOVERED = "recovered"


def frame_outcome(detected: bool, max_codeword_errors: int, n: int, k: int) -> FrameOutcome:
    """Classify a frame under RS(n, k): recovered iff detected and every codeword has <= (n-k)//2 errors."""
    if not detected:
        return FrameOutcome.DETECTION_FAILED
    return FrameOutcome.RECOVERED if max_codeword_errors <= (n - k) // 2 else FrameOutcome.NOT_RECOVERABLE


@dataclass(frozen=True)
class YieldBreakdown:
    n_frames: int
    detection_failed: int
    not_recoverable: int
    recovered: int

    @property
    def frame_yield(self) -> float:
        """Recovered / all code frames: the yield that enters goodput."""
        return self.recovered / self.n_frames if self.n_frames else 0.0

    def fractions(self) -> dict[str, float]:
        n = self.n_frames or 1
        return {o.value: getattr(self, o.value) / n for o in FrameOutcome}


def yield_breakdown(outcomes: list[FrameOutcome]) -> YieldBreakdown:
    counts = {o: sum(1 for x in outcomes if x == o) for o in FrameOutcome}
    return YieldBreakdown(len(outcomes), counts[FrameOutcome.DETECTION_FAILED], counts[FrameOutcome.NOT_RECOVERABLE],
                          counts[FrameOutcome.RECOVERED])


def goodput_bytes_per_s(payload_bytes_per_frame: float, frame_yield: float, fps: float = ASSUMED_FPS) -> float:
    """The documented goodput formula. Pure arithmetic, no timing."""
    if not 0.0 <= frame_yield <= 1.0:
        raise ValueError("frame_yield must be in [0, 1]")
    return payload_bytes_per_frame * frame_yield * fps


def goodput_mbit_per_s(payload_bytes_per_frame: float, frame_yield: float, fps: float = ASSUMED_FPS) -> float:
    return goodput_bytes_per_s(payload_bytes_per_frame, frame_yield, fps) * BITS_PER_BYTE / 1e6
