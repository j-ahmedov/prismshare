"""Reed-Solomon RS(n, k) over GF(2^8), with erasure support.

Thin wrapper around ``reedsolo`` with its defaults (primitive polynomial
0x11d, generator 2, first consecutive root 0). Codes with n < 255 are
shortened codes: the message is k bytes, the codeword k + (n - k) = n bytes.

A codeword with ``e`` symbol errors at unknown positions and ``f`` erasures
(errors at known positions) is guaranteed decodable iff ``2e + f <= n - k``.
``is_recoverable`` states that bound; ecc_sim.py (step 6) applies it to the
recorded error patterns instead of re-running the decoder.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import reedsolo

from prism_share.codec.params import RS_MAX_CODEWORD


def _check(n: int, k: int) -> None:
    if not 0 < k < n <= RS_MAX_CODEWORD:
        raise ValueError(f"need 0 < k < n <= {RS_MAX_CODEWORD}, got n={n}, k={k}")


@functools.lru_cache(maxsize=None)
def _codec(parity: int) -> reedsolo.RSCodec:
    return reedsolo.RSCodec(nsym=parity, nsize=RS_MAX_CODEWORD)


def rs_encode(message: bytes, n: int, k: int) -> bytes:
    """Systematic encoding: returns ``message`` (k bytes) followed by n - k parity bytes."""
    _check(n, k)
    if len(message) != k:
        raise ValueError(f"message must be {k} bytes, got {len(message)}")
    return bytes(_codec(n - k).encode(bytearray(message)))


def rs_decode(codeword: bytes, n: int, k: int, erasures: Sequence[int] = ()) -> bytes | None:
    """Decode an n-byte codeword; returns the k message bytes, or None on failure.

    ``erasures`` are symbol positions (0..n-1) known to be unreliable. Failure
    means the decoder detected an uncorrectable pattern; beyond the guaranteed
    bound it may also *miscorrect* silently, which is why every frame carries a
    CRC-32 on top of RS.
    """
    _check(n, k)
    if len(codeword) != n:
        raise ValueError(f"codeword must be {n} bytes, got {len(codeword)}")
    erase_pos = sorted(set(int(p) for p in erasures))
    if any(not 0 <= p < n for p in erase_pos):
        raise ValueError("erasure position out of range")
    if len(erase_pos) > n - k:
        # More erasures than parity: unusable as erasures, fall back to error decoding.
        erase_pos = []
    try:
        message, _, _ = _codec(n - k).decode(bytearray(codeword), erase_pos=erase_pos or None)
    except reedsolo.ReedSolomonError:
        return None
    return bytes(message)


def is_recoverable(n_errors: int, n_erasures: int, n: int, k: int) -> bool:
    """The RS guarantee: ``2 * errors + erasures <= n - k``."""
    _check(n, k)
    return 2 * n_errors + n_erasures <= n - k
