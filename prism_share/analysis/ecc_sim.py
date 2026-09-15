"""Simulate any RS(n, k) from a recorded byte error pattern (hard requirement 5).

At fixed grid and colour depth the cells, and therefore the physical frame
statistics, do not depend on the ECC rate; RS(n, k) only decides how the
frame's bytes are split into codewords and how many are parity. So a frame is
captured (or simulated) once, its byte error pattern against ground truth is
recorded, and here we compute what any RS(n, k) would have done:

* codewords of length n are formed exactly as framing.py interleaves them:
  n_codewords = capacity_bytes // n, symbol s of codeword j at stream byte
  s * n_codewords + j;
* a codeword is recoverable iff 2 * errors + erasures <= n - k (the RS
  guarantee; the decoder may occasionally do better, never assumed);
* a frame is recoverable iff every codeword is.

Only the *maximum* per-codeword error count of a frame matters for a given n,
so ``max_codeword_errors`` is the statistic stored per frame; a frame is then
recoverable for every k with (n - k) // 2 >= that maximum.

**The RS code is chosen out of sample** (``out_of_sample``). Choosing the
goodput-maximising (n, k) on the same frames it is scored on is an in-sample
optimum. It picks whichever code happened to fit those frames' worst codeword,
so it overstates goodput, and it overstates it most for configurations near
their yield cliff. More frames dilute that bias but never remove it. The frames
are therefore split by position (``params.RS_SELECTION_PERIOD``): the code is
chosen on the selection half and scored only on the evaluation half.
``best_code`` on all frames remains available, but only as a diagnostic of how
large the bias was.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prism_share.analysis.metrics import goodput_bytes_per_s
from prism_share.codec.framing import (
    CRC_BYTES,
    HEADER_BYTES,
    codeword_stream_positions,
    frame_capacity,
    frame_capacity_band_credited,
)
from prism_share.codec.params import ASSUMED_FPS, RS_MAX_CODEWORD, RS_SELECTION_PERIOD, CodecParams

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def codeword_error_counts(byte_errors: npt.ArrayLike, n: int) -> IntArray:
    """Symbol errors in each length-n codeword of one frame's byte error pattern."""
    errs = np.asarray(byte_errors, dtype=bool)
    if not 1 <= n <= RS_MAX_CODEWORD:
        raise ValueError(f"n must be in [1, {RS_MAX_CODEWORD}]")
    n_codewords = len(errs) // n
    if n_codewords == 0:
        return np.zeros(0, dtype=np.int64)
    return errs[codeword_stream_positions(n_codewords, n)].sum(axis=1).astype(np.int64)


def max_codeword_errors(byte_errors: npt.ArrayLike, n: int) -> int:
    counts = codeword_error_counts(byte_errors, n)
    return int(counts.max()) if len(counts) else 0


def frame_recoverable(byte_errors: npt.ArrayLike, n: int, k: int) -> bool:
    return max_codeword_errors(byte_errors, n) <= (n - k) // 2


def payload_bytes_per_frame(params: CodecParams, n: int, k: int, *, band_credited: bool = False) -> int:
    """Payload bytes one frame of ``params``'s grid carries under RS(n, k) (0 if header won't fit).

    ``band_credited`` counts the cells the index band displaces as if they
    carried data: the payload of a deployed codec with no band.
    """
    if not 0 < k < n <= RS_MAX_CODEWORD:
        raise ValueError(f"need 0 < k < n <= {RS_MAX_CODEWORD}")
    capacity = frame_capacity_band_credited(params) if band_credited else frame_capacity(params)
    n_codewords = capacity.capacity_bytes // n
    return max(0, n_codewords * k - HEADER_BYTES - CRC_BYTES)


@dataclass(frozen=True)
class EccOutcome:
    n: int
    k: int
    frame_yield: float
    payload_bytes: int
    goodput_bytes_per_s: float


def evaluate(max_errors_per_frame: Sequence[int], params: CodecParams, n: int, k: int, fps: float = ASSUMED_FPS) -> EccOutcome:
    """Yield and goodput of RS(n, k) over frames summarised by their max codeword error count."""
    worst = np.asarray(max_errors_per_frame, dtype=np.int64)
    if not len(worst):
        raise ValueError("no frames")
    frame_yield = float(np.mean(worst <= (n - k) // 2))
    payload = payload_bytes_per_frame(params, n, k)
    return EccOutcome(n, k, frame_yield, payload, goodput_bytes_per_s(payload, frame_yield, fps))


def best_rate(max_errors_per_frame: Sequence[int], params: CodecParams, n: int, fps: float = ASSUMED_FPS) -> EccOutcome:
    """The k in [1, n-1] maximising goodput for codeword length n on these frames (ties -> larger k).

    Scored on the frames it was chosen on, this is an in-sample optimum: use
    ``out_of_sample`` for any goodput that is reported.
    """
    outcomes = [evaluate(max_errors_per_frame, params, n, k, fps) for k in range(n - 1, 0, -1)]
    return max(outcomes, key=lambda o: (o.goodput_bytes_per_s, o.k))


def best_code(max_errors_by_n: Mapping[int, Sequence[int]], params: CodecParams, fps: float = ASSUMED_FPS) -> EccOutcome:
    """The goodput-maximising RS(n, k) over every codeword length given, on these frames.

    Ties go to the higher code rate. In-sample by construction: see ``out_of_sample``.
    """
    return max((best_rate(worst, params, n, fps) for n, worst in max_errors_by_n.items()),
               key=lambda o: (o.goodput_bytes_per_s, o.k / o.n))


def selection_mask(positions: npt.ArrayLike) -> BoolArray:
    """True for frames in the selection half, False for the evaluation half (by position only)."""
    return np.asarray(positions, dtype=np.int64) % RS_SELECTION_PERIOD == 0


@dataclass(frozen=True)
class OutOfSample:
    """An RS code chosen on the selection half and scored on the evaluation half."""

    chosen: EccOutcome  # the choice, with its (in-sample) score on the selection half
    evaluated: EccOutcome  # the same (n, k) scored on the evaluation half: the reported number
    n_selection: int
    n_evaluation: int


def out_of_sample(
    max_errors_by_n: Mapping[int, Sequence[int]], positions: npt.ArrayLike, params: CodecParams, fps: float = ASSUMED_FPS
) -> OutOfSample:
    """Choose RS(n, k) on the selection half of the frames, score it on the evaluation half.

    ``max_errors_by_n[n][i]`` is frame i's maximum codeword error count for
    length n, and ``positions[i]`` is its position (see RS_SELECTION_PERIOD).
    Both halves must be non-empty: a goodput with no held-out frames is not
    reported at all, rather than silently reported in sample.
    """
    select = selection_mask(positions)
    if not select.any() or select.all():
        raise ValueError(f"out-of-sample RS selection needs frames in both halves; got {int(select.sum())} selection "
                         f"and {int((~select).sum())} evaluation frames")
    arrays = {n: np.asarray(worst, dtype=np.int64) for n, worst in max_errors_by_n.items()}
    if any(len(a) != len(select) for a in arrays.values()):
        raise ValueError("positions and error counts differ in length")
    chosen = best_code({n: a[select] for n, a in arrays.items()}, params, fps)
    evaluated = evaluate(arrays[chosen.n][~select], params, chosen.n, chosen.k, fps)
    return OutOfSample(chosen, evaluated, int(select.sum()), int((~select).sum()))
