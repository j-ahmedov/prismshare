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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prism_share.analysis.metrics import goodput_bytes_per_s
from prism_share.codec.framing import CRC_BYTES, HEADER_BYTES, codeword_stream_positions, frame_capacity
from prism_share.codec.params import ASSUMED_FPS, RS_MAX_CODEWORD, CodecParams

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


def payload_bytes_per_frame(params: CodecParams, n: int, k: int) -> int:
    """Payload bytes one frame of ``params``'s grid carries under RS(n, k) (0 if header won't fit)."""
    if not 0 < k < n <= RS_MAX_CODEWORD:
        raise ValueError(f"need 0 < k < n <= {RS_MAX_CODEWORD}")
    n_codewords = frame_capacity(params).capacity_bytes // n
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
    """The k in [1, n-1] maximising goodput for codeword length n (ties -> larger k).

    Note: when k is chosen on the same frames it is evaluated on, the result is
    an in-sample optimum, slightly optimistic for small frame counts.
    """
    outcomes = [evaluate(max_errors_per_frame, params, n, k, fps) for k in range(n - 1, 0, -1)]
    return max(outcomes, key=lambda o: (o.goodput_bytes_per_s, o.k))
