"""Fountain (rateless erasure) code over frames.

The payload is split into K source blocks of B bytes (the last zero-padded).
Every frame carries one *encoded block* identified by ``block_id``:

* block_id < K: the source block itself (systematic part);
* block_id >= K: the XOR of a pseudo-random subset of source blocks. The subset
  is a K-bit coefficient vector drawn from the SHA-256 keystream seeded by
  CodecParams.seed and indexed by block_id; an all-zero draw is replaced by the
  unit vector of block ``block_id mod K``.

This is a dense random linear fountain over GF(2) decoded by Gaussian
elimination, rather than an LT code with a peeling decoder. For the tens to
hundreds of blocks a thesis-sized payload produces, LT needs a large reception
overhead, whereas a dense code recovers the payload from any set of received
blocks whose coefficient vectors have rank K; the chance that K + m received
blocks fall short decays roughly as 2^-m. That is close to the ideal erasure code the goodput formula assumes (see
README, "Goodput"). Decoding is O(K^2 * frames), irrelevant offline.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from prism_share.codec.params import FOUNTAIN_MIN_REPAIR, FOUNTAIN_REPAIR_FRACTION
from prism_share.codec.prng import keystream_bits

UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


def n_source_blocks(payload_len: int, block_bytes: int) -> int:
    """K: number of source blocks (at least one, even for an empty payload)."""
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    return max(1, -(-payload_len // block_bytes))


def default_frame_count(n_source: int) -> int:
    """Source blocks plus the default number of repair blocks."""
    repair = max(FOUNTAIN_MIN_REPAIR, int(np.ceil(n_source * FOUNTAIN_REPAIR_FRACTION)))
    return n_source + repair


def split_payload(payload: bytes, block_bytes: int) -> UInt8Array:
    """(K, block_bytes) array of zero-padded source blocks."""
    k = n_source_blocks(len(payload), block_bytes)
    blocks = np.zeros(k * block_bytes, dtype=np.uint8)
    blocks[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return blocks.reshape(k, block_bytes)


def coefficients(block_id: int, n_source: int, seed: int) -> BoolArray:
    """GF(2) coefficient vector (length K) of encoded block ``block_id``."""
    if block_id < 0:
        raise ValueError("block_id must be >= 0")
    coeffs = np.zeros(n_source, dtype=bool)
    if block_id < n_source:
        coeffs[block_id] = True
        return coeffs
    coeffs = keystream_bits(seed, f"fountain-k{n_source}", n_source, index=block_id).copy()
    if not coeffs.any():
        coeffs[block_id % n_source] = True
    return coeffs


def encode_block(source: UInt8Array, block_id: int, seed: int) -> bytes:
    """Encoded block ``block_id``: XOR of the source blocks selected by its coefficients."""
    coeffs = coefficients(block_id, len(source), seed)
    return np.bitwise_xor.reduce(source[coeffs], axis=0).tobytes()


def decode_blocks(
    received: Mapping[int, bytes], n_source: int, block_bytes: int, seed: int
) -> UInt8Array | None:
    """Recover the (K, block_bytes) source blocks, or None if rank < K.

    ``received`` maps block_id to that block's bytes (duplicates are irrelevant:
    a mapping holds one entry per id).
    """
    if not received:
        return None
    ids = sorted(received)
    matrix = np.array([coefficients(i, n_source, seed) for i in ids], dtype=bool)
    data = np.array([np.frombuffer(received[i], dtype=np.uint8) for i in ids], dtype=np.uint8)
    if data.shape[1] != block_bytes:
        raise ValueError(f"blocks must be {block_bytes} bytes")

    pivot_row = 0
    for col in range(n_source):
        candidates = np.flatnonzero(matrix[pivot_row:, col])
        if not len(candidates):
            return None  # column without a pivot: rank < K
        r = pivot_row + int(candidates[0])
        if r != pivot_row:
            matrix[[pivot_row, r]] = matrix[[r, pivot_row]]
            data[[pivot_row, r]] = data[[r, pivot_row]]
        others = matrix[:, col].copy()
        others[pivot_row] = False
        matrix[others] ^= matrix[pivot_row]
        data[others] ^= data[pivot_row]
        pivot_row += 1
    # Fully reduced: row i is the unit vector e_i.
    return data[:n_source]
